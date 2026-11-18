#pragma once

#include <cstddef>
#include <cstdint>
#include <span>

namespace tiny_llm {

/**
 * @brief Size of the llama2.c checkpoint header: 7 `int32` fields, 28 bytes.
 *
 * A fact of the file format, not of any model: parse_config reads these bytes and
 * map_weights starts right after them, so both share this one definition.
 */
inline constexpr std::size_t header_size_in_bytes = 7 * sizeof(std::int32_t);

/**
 * @brief The model hyperparameters stored in the header of a llama2.c checkpoint.
 *
 * Plain data with no invariants of its own: parse_config() validates the values.
 * `sizeof(Config)` is 64 bytes: seven `size_t` (56) and a `bool`, padded to a multiple of 8.
 */
struct Config {
    std::size_t embedding_size = 0;         /**< Width of each token vector (`dim` in llama2.c). */
    std::size_t feed_forward_size = 0;      /**< Hidden width of the feed-forward network. */
    std::size_t layer_count = 0;            /**< Number of transformer layers. */
    std::size_t head_count = 0;             /**< Number of query heads. */
    std::size_t kv_head_count = 0;          /**< Number of key/value heads (fewer with GQA). */
    std::size_t vocab_size = 0;             /**< Number of tokens in the vocabulary. */
    std::size_t max_sequence_length = 0;    /**< Longest sequence the model was trained for. */
    bool shares_classifier_weights = false; /**< Classifier reuses the embedding table. */
};

/**
 * @brief Reads the 28-byte header of a llama2.c checkpoint (7 little-endian `int32`).
 *
 * A negative vocabulary size in the file means the classifier has its own weights.
 *
 * @param bytes The checkpoint bytes; only the first 28 are read. Typically MappedFile::bytes().
 * @return The validated hyperparameters.
 * @throws LoadError If there are fewer than 28 bytes, a size is not positive, the embedding
 *         size is not divisible by the head count, or the head count by the K/V head count.
 *         The message does not include the path: the caller adds it.
 */
[[nodiscard]] Config parse_config(std::span<const std::byte> bytes);

} // namespace tiny_llm
