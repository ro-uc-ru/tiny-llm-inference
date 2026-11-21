#include <tiny_llm/check.hpp>
#include <tiny_llm/kernels.hpp>
#include <tiny_llm/transformer.hpp>

#include <cstddef>
#include <format>
#include <span>

namespace tiny_llm {

Transformer::Transformer(const std::filesystem::path& checkpoint_path)
    : file_{checkpoint_path}, config_{parse_config(file_.bytes())},
      weights_{map_weights(file_.bytes(), config_)}, state_{make_run_state(config_)} {}

std::size_t Transformer::cache_offset(std::size_t layer, std::size_t position) const noexcept {
    TINY_LLM_CHECK(layer < config_.layer_count,
                   std::format("layer {} is out of range, the model has {} layers", layer,
                               config_.layer_count));
    TINY_LLM_CHECK(position < config_.max_sequence_length,
                   std::format("position {} is out of range, the maximum sequence length is {}",
                               position, config_.max_sequence_length));
    const std::size_t head_size = config_.embedding_size / config_.head_count;
    return (layer * config_.max_sequence_length + position) * head_size * config_.kv_head_count;
}

std::span<const float> Transformer::forward(std::size_t token, std::size_t position) {
    // 1. Checks: a token or position out of range is a bug in the caller (sampler or main).
    TINY_LLM_CHECK(token < config_.vocab_size,
                   std::format("token {} is out of range, the vocabulary has {} tokens", token,
                               config_.vocab_size));
    TINY_LLM_CHECK(position < config_.max_sequence_length,
                   std::format("position {} is out of range, the maximum sequence length is {}",
                               position, config_.max_sequence_length));

    // 2. Sizes used in the whole method.
    const std::size_t head_size = config_.embedding_size / config_.head_count; // d_h
    const std::size_t kv_size = config_.kv_head_count * head_size;             // H_kv · d_h
    const std::size_t query_heads_per_kv_head = config_.head_count / config_.kv_head_count;
    const std::size_t position_count = position + 1; // attend to 0 … position, both included

    // Views of the whole caches, so they can be cut with subspan (a vector has no subspan).
    const std::span<float> key_cache{state_.key_cache};
    const std::span<float> value_cache{state_.value_cache};

    // 3. hidden = row `token` of the embedding table. Copied: hidden is modified below,
    //    and the row is read-only, inside the mapped file.
    const std::span<const float> embedding_row = weights_.token_embedding.row(token);
    for (std::size_t index = 0; index < embedding_row.size(); index++) {
        state_.hidden[index] = embedding_row[index];
    }

    // 4. The layers. The output of one is the input of the next, through state_.hidden.
    for (std::size_t layer = 0; layer < config_.layer_count; layer++) {
        // 4a. Attention. RMSNorm first: hidden itself is not touched until the residual.
        rmsnorm(state_.normalized, state_.hidden, weights_.attention_norm[layer].data());

        // Query into its buffer. Key and value straight into the caches, at this layer and
        // position: no intermediate buffer, and no copy afterwards.
        const std::size_t offset = cache_offset(layer, position);
        const std::span<float> key = key_cache.subspan(offset, kv_size);
        const std::span<float> value = value_cache.subspan(offset, kv_size);
        matmul(state_.query, weights_.query_projection[layer], state_.normalized);
        matmul(key, weights_.key_projection[layer], state_.normalized);
        matmul(value, weights_.value_projection[layer], state_.normalized);

        // RoPE turns the query and the new key by the angle of this position. The value is
        // not turned. The keys of earlier positions were turned when they were written.
        rotate_rope(state_.query, position, head_size);
        rotate_rope(key, position, head_size);

        // Every key and value of this layer, from position 0 on.
        const std::size_t layer_offset = cache_offset(layer, 0);
        const std::span<const float> layer_keys = key_cache.subspan(layer_offset);
        const std::span<const float> layer_values = value_cache.subspan(layer_offset);

        // One attention per query head; with GQA, a group of query heads shares a KV head.
        const std::span<const float> query{state_.query};
        const std::span<float> attention_result{state_.attention_result};
        const std::span<float> attention_scores{state_.attention_scores};
        for (std::size_t head = 0; head < config_.head_count; head++) {
            const std::size_t kv_head = head / query_heads_per_kv_head;
            attend(attention_result.subspan(head * head_size, head_size),
                   query.subspan(head * head_size, head_size),
                   layer_keys.subspan(kv_head * head_size),
                   layer_values.subspan(kv_head * head_size),
                   attention_scores.subspan(head * config_.max_sequence_length,
                                            config_.max_sequence_length),
                   position_count, kv_size);
        }

        // Mix the heads with Wo and add the result to the residual stream.
        matmul(state_.projection, weights_.attention_output[layer], state_.attention_result);
        add_in_place(state_.hidden, state_.projection);

        // 4b. Feed-forward network: w2 · (silu(w1 · x) * (w3 · x)), with x = RMSNorm(hidden).
        rmsnorm(state_.normalized, state_.hidden, weights_.ffn_norm[layer].data());
        matmul(state_.feed_forward_hidden, weights_.feed_forward_hidden[layer], state_.normalized);
        matmul(state_.feed_forward_gate, weights_.feed_forward_gate[layer], state_.normalized);
        swiglu(state_.feed_forward_hidden, state_.feed_forward_gate);
        matmul(state_.projection, weights_.feed_forward_output[layer], state_.feed_forward_hidden);
        add_in_place(state_.hidden, state_.projection);
    }

    // 5. Final RMSNorm and the classifier: one score per token of the vocabulary.
    rmsnorm(state_.normalized, state_.hidden, weights_.final_norm.data());
    matmul(state_.logits, weights_.classifier, state_.normalized);

    // A read-only view of state_.logits: valid until the next call overwrites it.
    return state_.logits;
}

} // namespace tiny_llm
