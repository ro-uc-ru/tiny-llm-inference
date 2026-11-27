#include <tiny_llm/options.hpp>

#include <cctype>      // std::isdigit
#include <cstddef>     // std::size_t
#include <cstdint>     // std::uint64_t
#include <format>      // std::format
#include <span>        // std::span
#include <stdexcept>   // std::invalid_argument
#include <string>      // std::string, std::stof, std::stoul
#include <string_view> // std::string_view

namespace tiny_llm {

/**
 * @brief std::stof, plus a check that it read the whole text.
 *
 * @param flag The flag the value belongs to, for the error message.
 * @param text The value as written on the command line.
 * @return The float `text` spells.
 * @throws std::invalid_argument If `text` is not a number or has characters left over.
 * @throws std::out_of_range     If it does not fit in a float.
 */
float parse_float(std::string_view flag, const std::string& text) {
    // stof writes how many characters it read, like strtof's end pointer. If `text` does not
    // start with a number, it throws std::invalid_argument itself ("stof: no conversion").
    std::size_t parsed_count = 0;
    const float value = std::stof(text, &parsed_count);
    // Characters left over mean a typo such as "0.9x", which stof alone would accept as 0.9.
    if (parsed_count != text.size()) {
        throw std::invalid_argument{std::format("{} expects a number, got {}", flag, text)};
    }
    return value;
}

/**
 * @brief std::stoul, plus a leading digit and a check that it read the whole text.
 *
 * @param flag The flag the value belongs to, for the error message.
 * @param text The value as written on the command line.
 * @return The integer `text` spells.
 * @throws std::invalid_argument If `text` is not a non-negative integer.
 * @throws std::out_of_range     If it does not fit in 64 bits.
 */
std::uint64_t parse_count(std::string_view flag, const std::string& text) {
    const auto message = std::format("{} expects a non-negative integer, got {}", flag, text);
    // stoul skips leading spaces and accepts a sign: "-5" and " -5" wrap around to 2^64 - 5.
    // So the text must start with a digit. empty() goes first: an empty text has no text[0].
    // isdigit takes an unsigned char, like isprint: a negative char is undefined behaviour.
    if (text.empty() || std::isdigit(static_cast<unsigned char>(text[0])) == 0) {
        throw std::invalid_argument{message};
    }
    // As in parse_float: characters left over mean a typo such as "1e9", which stoul reads as 1.
    std::size_t parsed_count = 0;
    const std::uint64_t value = std::stoul(text, &parsed_count);
    if (parsed_count != text.size()) {
        throw std::invalid_argument{message};
    }
    return value;
}

/**
 * @brief The checkpoint, then flag and value pairs, then run.c's two corrections.
 *
 * @param arguments The whole argv, program name included.
 * @return The options, with run.c's defaults for every flag not given.
 * @throws std::invalid_argument On a missing checkpoint, an unknown flag, a flag without a
 *                               value or a badly written number.
 * @throws std::out_of_range     On a number too large for its type.
 */
Options parse_options(std::span<char*> arguments) {
    // arguments[0] is the program name; the checkpoint is the only argument without a flag.
    if (arguments.size() < 2) {
        throw std::invalid_argument{"missing checkpoint path"};
    }
    Options options{};
    options.checkpoint_path = arguments[1];

    for (std::size_t index = 2; index < arguments.size(); index += 2) {
        // A string_view, so that == compares text; with the raw char* it compares addresses.
        const std::string_view flag{arguments[index]};
        // Checked before reading arguments[index + 1], which would be past the end of the span.
        if (index + 1 >= arguments.size()) {
            throw std::invalid_argument{std::format("{} needs a value", flag)};
        }
        // A std::string, not a string_view: std::stof and std::stoul only take std::string.
        const std::string value{arguments[index + 1]};
        if (flag == "-t") {
            options.temperature = parse_float(flag, value);
        } else if (flag == "-p") {
            options.top_p = parse_float(flag, value);
        } else if (flag == "-n") {
            options.steps = parse_count(flag, value);
        } else if (flag == "-s") {
            options.seed = parse_count(flag, value);
        } else if (flag == "-i") {
            options.prompt = value;
        } else if (flag == "-z") {
            options.tokenizer_path = value;
        } else {
            throw std::invalid_argument{std::format("unknown flag {}", flag)};
        }
    }

    // run.c's corrections, so that the same command line samples the same way.
    if (options.temperature < 0.0f) {
        options.temperature = 0.0f;
    }
    if (options.top_p < 0.0f || options.top_p > 1.0f) {
        options.top_p = 0.9f;
    }
    return options;
}

} // namespace tiny_llm
