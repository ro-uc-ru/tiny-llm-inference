#pragma once

#include <tiny_llm/config.hpp>

#include <vector>

namespace tiny_llm {

/**
 * @brief Every buffer the forward pass writes, allocated once. Owns all of them.
 *
 * Built by make_run_state and reused for every token: the forward pass overwrites these
 * buffers and never allocates. Sizes use head_size = embedding_size / head_count, as in
 * map_weights. Almost all the memory is the two caches (75.5 MB of 75.7 MB in stories110M).
 */
struct RunState {
    std::vector<float> hidden;              /**< (embedding_size) the residual stream. */
    std::vector<float> normalized;          /**< (embedding_size) hidden after RMSNorm. */
    std::vector<float> query;               /**< (embedding_size) query of the current token. */
    std::vector<float> attention_result;    /**< (embedding_size) attention, before Wo. */
    std::vector<float> projection;          /**< (embedding_size) output of Wo and of w2. */
    std::vector<float> feed_forward_hidden; /**< (feed_forward_size) output of w1. */
    std::vector<float> feed_forward_gate;   /**< (feed_forward_size) output of w3. */
    std::vector<float> attention_scores;    /**< (head_count * max_sequence_length). */
    std::vector<float> logits;              /**< (vocab_size). */
    /** (layer_count * max_sequence_length * kv_head_count * head_size), one flat buffer. */
    std::vector<float> key_cache;
    /** Same size and layout as key_cache. */
    std::vector<float> value_cache;
};

/**
 * @brief Allocates every buffer of the forward pass for this config.
 *
 * @param config A config already validated by parse_config (head_count divides embedding_size).
 * @return A RunState with every buffer sized for this config and set to zero.
 */
[[nodiscard]] RunState make_run_state(const Config& config);

} // namespace tiny_llm
