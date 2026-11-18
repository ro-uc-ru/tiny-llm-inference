#pragma once

#include <tiny_llm/config.hpp>
#include <tiny_llm/tensor.hpp>

#include <cstddef>
#include <span>
#include <vector>

namespace tiny_llm {

/**
 * @brief Named views over the weight section of a llama2.c checkpoint.
 *
 * Members are in file order. Each vector holds one Tensor per layer, shaped like that
 * layer's matrix. Weights owns no data: every Tensor looks into the bytes that
 * map_weights received, so the owner of those bytes (the MappedFile, held by the
 * Transformer) must outlive it.
 */
struct Weights {
    Tensor token_embedding;                  /**< (vocab_size, embedding_size). */
    std::vector<Tensor> attention_norm;      /**< (embedding_size, 1), one per layer. */
    std::vector<Tensor> query_projection;    /**< (head_count * head_size, embedding_size). */
    std::vector<Tensor> key_projection;      /**< (kv_head_count * head_size, embedding_size). */
    std::vector<Tensor> value_projection;    /**< (kv_head_count * head_size, embedding_size). */
    std::vector<Tensor> attention_output;    /**< (embedding_size, embedding_size). */
    std::vector<Tensor> ffn_norm;            /**< (embedding_size, 1). */
    std::vector<Tensor> feed_forward_hidden; /**< w1: (feed_forward_size, embedding_size). */
    std::vector<Tensor> feed_forward_output; /**< w2: (embedding_size, feed_forward_size). */
    std::vector<Tensor> feed_forward_gate;   /**< w3: (feed_forward_size, embedding_size). */
    Tensor final_norm;                       /**< (embedding_size, 1). */
    Tensor classifier; /**< (vocab_size, embedding_size); may view token_embedding's data. */
};

/**
 * @brief How many floats follow the header, for a checkpoint with this config.
 *
 * Counts every matrix in the file, plus the RoPE tables that llama2.c still stores
 * (max_sequence_length * head_size floats, never read), plus the classifier when it is
 * not shared with the embedding.
 *
 * @param config A config already validated by parse_config.
 * @return The number of floats after the header; the file must be
 *         `header_size_in_bytes + sizeof(float) * weight_float_count(config)` bytes.
 */
[[nodiscard]] std::size_t weight_float_count(const Config& config);

/**
 * @brief Slices the weight section of a llama2.c checkpoint into named views.
 *
 * @param file_bytes The whole file, header included. Must outlive the result.
 * @param config     The header, already parsed by parse_config.
 * @return One Tensor per matrix; none of them owns its data.
 * @throws LoadError If the file size is not
 *         `header_size_in_bytes + sizeof(float) * weight_float_count(config)`.
 */
[[nodiscard]] Weights map_weights(std::span<const std::byte> file_bytes, const Config& config);

} // namespace tiny_llm
