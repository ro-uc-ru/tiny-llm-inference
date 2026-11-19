#include <tiny_llm/run_state.hpp>

#include <cstddef>
#include <vector>

namespace tiny_llm {

RunState make_run_state(const Config& config) {
    const std::size_t head_size = config.embedding_size / config.head_count;
    const std::size_t cache_size =
        config.layer_count * config.max_sequence_length * config.kv_head_count * head_size;

    // Parentheses, not braces: std::vector<float>(n) is n zeros, std::vector<float>{n} is
    // one element equal to n.
    RunState state;
    state.hidden = std::vector<float>(config.embedding_size);
    state.normalized = std::vector<float>(config.embedding_size);
    state.query = std::vector<float>(config.embedding_size);
    state.attention_result = std::vector<float>(config.embedding_size);
    state.projection = std::vector<float>(config.embedding_size);
    state.feed_forward_hidden = std::vector<float>(config.feed_forward_size);
    state.feed_forward_gate = std::vector<float>(config.feed_forward_size);
    state.attention_scores = std::vector<float>(config.head_count * config.max_sequence_length);
    state.logits = std::vector<float>(config.vocab_size);
    state.key_cache = std::vector<float>(cache_size);
    state.value_cache = std::vector<float>(cache_size);
    return state;
}

} // namespace tiny_llm
