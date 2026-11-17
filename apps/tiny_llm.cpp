#include <tiny_llm/config.hpp>
#include <tiny_llm/load_error.hpp>
#include <tiny_llm/mapped_file.hpp>

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <print>

namespace {

/**
 * @brief Prints the hyperparameters read from the checkpoint header, one per line.
 * @param config The parsed header.
 */
void print_config(const tiny_llm::Config& config) {
    std::println("embedding_size:            {}", config.embedding_size);
    std::println("feed_forward_size:         {}", config.feed_forward_size);
    std::println("layer_count:               {}", config.layer_count);
    std::println("head_count:                {}", config.head_count);
    std::println("kv_head_count:             {}", config.kv_head_count);
    std::println("vocab_size:                {}", config.vocab_size);
    std::println("max_sequence_length:       {}", config.max_sequence_length);
    std::println("shares_classifier_weights: {}", config.shares_classifier_weights);
}

} // namespace

/**
 * @brief Maps a llama2.c checkpoint and prints its header.
 *
 * The only place that catches LoadError: it adds the path, which the thrower may not know,
 * prints the error and exits with 1. Any other exception is unexpected and is left to
 * std::terminate, like a bug. See docs/error-handling.md.
 *
 * Exit codes: 0 on success, 1 for a load error, 2 for a usage error.
 */
int main(int argument_count, char** arguments) {
    if (argument_count != 2) {
        std::println(stderr, "usage: tiny_llm <checkpoint.bin>");
        return 2;
    }
    const std::filesystem::path checkpoint_path{arguments[1]};

    try {
        // The mapping must outlive every view of its bytes: here, until the end of the try.
        const tiny_llm::MappedFile checkpoint{checkpoint_path};
        print_config(tiny_llm::parse_config(checkpoint.bytes()));
    } catch (const tiny_llm::LoadError& error) {
        std::println(stderr, "error loading {}: {}", checkpoint_path.string(), error.what());
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
