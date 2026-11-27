#include <tiny_llm/load_error.hpp>
#include <tiny_llm/options.hpp>
#include <tiny_llm/sampler.hpp>
#include <tiny_llm/tokenizer.hpp>
#include <tiny_llm/transformer.hpp>

#include <cctype>      // std::isprint, std::isspace
#include <chrono>      // std::chrono::steady_clock, std::chrono::duration
#include <cstddef>     // std::size_t
#include <cstdint>     // std::uint64_t
#include <cstdio>      // stderr
#include <cstdlib>     // EXIT_SUCCESS, EXIT_FAILURE
#include <ctime>       // std::time
#include <print>       // std::print, std::println
#include <span>        // std::span
#include <stdexcept>   // std::invalid_argument, std::out_of_range
#include <string_view> // std::string_view
#include <vector>      // std::vector

namespace {

/** @brief Exit code for a badly written command line, as in the paso 0 main. */
constexpr int usage_error_exit_code = 2;

/**
 * @brief Prints `piece` unless it is a single byte that is neither printable nor whitespace.
 *
 * The same rule as run.c's safe_printf, so that both programs print the same bytes. A piece of
 * several bytes is always printed, even if it is part of a UTF-8 character.
 *
 * @param piece The text of one token, from Tokenizer::decode.
 */
void print_piece(std::string_view piece) {
    if (piece.size() == 1) {
        // isprint and isspace take an unsigned char: a char >= 128 is negative on Apple Silicon,
        // and passing a negative value is undefined behaviour.
        const auto byte = static_cast<unsigned char>(piece[0]);
        // Both return an int: 0 means no. A control byte such as 7 (the bell) is skipped;
        // '\n' is whitespace and is printed.
        if (std::isprint(byte) == 0 && std::isspace(byte) == 0) {
            return;
        }
    }
    std::print("{}", piece);
}

/**
 * @brief Runs the prompt through the model and samples until `steps` positions or the begin token.
 *
 * Prints the text to stdout as it goes, then a newline, and the tokens per second to stderr,
 * both as run.c does. The timer starts after position 0, whose time includes paging in the
 * mapped weights. Copies the logits of each position before sampling, because sample()
 * overwrites them with probabilities and forward() only lends them for reading.
 *
 * @param transformer The model; its caches fill up position by position.
 * @param tokenizer   Turns the prompt into tokens and each token back into text.
 * @param sampler     Chooses each token after the prompt.
 * @param prompt      The text to continue; may be empty.
 * @param steps       Positions to run, prompt included; 0 or above max_sequence_length means
 *                    max_sequence_length.
 */
void generate(tiny_llm::Transformer& transformer, const tiny_llm::Tokenizer& tokenizer,
              tiny_llm::Sampler& sampler, std::string_view prompt, std::size_t steps) {
    const auto& config = transformer.config();
    // Without this, a position would reach max_sequence_length and forward() would abort.
    if (steps == 0 || steps > config.max_sequence_length) {
        steps = config.max_sequence_length;
    }

    // Never empty: it starts with the begin token, even for an empty prompt.
    const auto prompt_tokens = tokenizer.encode(prompt, true);

    // Decision 2: our own copy of the logits, for sample() to write into. Allocated once,
    // before the loop: vocab_size floats, 32 000 × 4 = 128 000 bytes for stories110M.
    std::vector<float> logits(config.vocab_size);

    std::size_t token = prompt_tokens[0];
    // Set again after position 0; declared out here because it is read after the loop.
    auto start = std::chrono::steady_clock::now();
    // Also out here: after the loop it says how many positions ran.
    std::size_t position = 0;

    for (; position < steps; ++position) {
        // A view into the Transformer's state, valid only until the next forward().
        const std::span<const float> scores = transformer.forward(token, position);
        for (std::size_t index = 0; index < scores.size(); ++index) {
            logits[index] = scores[index];
        }

        // Prefill: while the prompt lasts, the next token is the prompt's, and the logits of
        // this position are not used. Then the sampler chooses.
        const std::size_t next = position + 1 < prompt_tokens.size() ? prompt_tokens[position + 1]
                                                                     : sampler.sample(logits);

        // The model ends a text with the begin token: the start of the next one.
        if (next == tiny_llm::begin_token) {
            break;
        }
        // decode needs the previous token: after the begin token, the leading space is dropped.
        print_piece(tokenizer.decode(token, next));
        token = next;

        if (position == 0) {
            start = std::chrono::steady_clock::now();
        }
    }
    std::println();

    // position − 1 positions ran after the timer started; with 0 or 1 there is nothing to time.
    if (position > 1) {
        const std::chrono::duration<double> elapsed = std::chrono::steady_clock::now() - start;
        std::println(stderr, "achieved tok/s: {}",
                     static_cast<double>(position - 1) / elapsed.count());
    }
}

} // namespace

/**
 * @brief Generates text from a llama2.c checkpoint, with run.c's command line.
 *
 * The only place that catches exceptions. LoadError is a problem with the files (exit 1);
 * std::invalid_argument and std::out_of_range come from a badly written command line (exit 2).
 * Any other exception is unexpected and is left to std::terminate, like a bug.
 * See docs/error-handling.md.
 *
 * @param argument_count The number of command-line arguments, program name included.
 * @param arguments      The arguments, as C passes them.
 * @return 0 on success, 1 for a load error, 2 for a usage error.
 */
int main(int argument_count, char** arguments) {
    // The same pointer and count as in C, now with a size() that travels with them.
    const std::span<char*> argument_view{arguments, static_cast<std::size_t>(argument_count)};

    try {
        // Not const: the seed may still come from the clock.
        tiny_llm::Options options = tiny_llm::parse_options(argument_view);
        // As run.c: no -s means a new story each time. time() is never 0, which the Sampler
        // checks, because a xorshift of 0 stays 0 forever.
        if (options.seed == 0) {
            options.seed = static_cast<std::uint64_t>(std::time(nullptr));
        }

        // The Tokenizer and the Sampler need vocab_size, which only the checkpoint knows, so the
        // Transformer goes first. All three live until the end of the try, and are destroyed in
        // reverse order.
        tiny_llm::Transformer transformer{options.checkpoint_path};
        const tiny_llm::Tokenizer tokenizer{options.tokenizer_path,
                                            transformer.config().vocab_size};
        tiny_llm::Sampler sampler{transformer.config().vocab_size, options.temperature,
                                  options.top_p, options.seed};

        generate(transformer, tokenizer, sampler, options.prompt, options.steps);
    } catch (const tiny_llm::LoadError& error) {
        std::println(stderr, "error: {}", error.what());
        return EXIT_FAILURE;
    } catch (const std::invalid_argument& error) {
        std::println(stderr, "error: {}\n{}", error.what(), tiny_llm::usage);
        return usage_error_exit_code;
    } catch (const std::out_of_range& error) {
        std::println(stderr, "error: {}\n{}", error.what(), tiny_llm::usage);
        return usage_error_exit_code;
    }
    return EXIT_SUCCESS;
}
