#pragma once

#include <cstddef>
#include <span>

namespace tiny_llm {

struct Config {
    std::size_t embedding_size = 0;
    std::size_t feed_forward_size = 0;
    std::size_t layer_count = 0;
    std::size_t head_count = 0;
    std::size_t kv_head_count = 0;
    std::size_t vocab_size = 0;
    std::size_t max_sequence_length = 0;
    bool shares_classifier_weights = false;
};

[[nodiscard]] Config parse_config(std::span<const std::byte> bytes);

} // namespace tiny_llm