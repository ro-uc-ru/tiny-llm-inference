#pragma once

#include <tiny_llm/config.hpp>
#include <tiny_llm/mapped_file.hpp>
#include <tiny_llm/run_state.hpp>
#include <tiny_llm/weights.hpp>

#include <cstddef>
#include <filesystem>
#include <span>

namespace tiny_llm {

/**
 * @brief A loaded llama2.c model: the mapped file, its header, its weights and its buffers.
 *
 * Declares no copy, move or destructor (rule of zero): the compiler builds them from the
 * members. It moves, because the mapping keeps its address and the weight views stay valid;
 * it does not copy, because MappedFile does not, and copied views would point into the
 * original's file.
 */
class Transformer {
  public:
    /**
     * @brief Maps the checkpoint, parses its header, slices its weights and allocates the buffers.
     *
     * Members are built in declaration order; if one throws, those already built are destroyed.
     *
     * @param checkpoint_path Path to a llama2.c `.bin` checkpoint.
     * @throws LoadError If the file cannot be mapped or does not match its header.
     */
    explicit Transformer(const std::filesystem::path& checkpoint_path);

    /** @brief The model hyperparameters, valid while the Transformer lives. */
    [[nodiscard]] const Config& config() const noexcept { return config_; }

    /**
     * @brief Runs one token through the model and returns the scores of the next token.
     *
     * Writes the key and value of this token into the caches of every layer, so positions
     * must come in order: 0, 1, 2… Each call reads what the earlier ones left there.
     * A token or position out of range is a bug in the caller and aborts with TINY_LLM_CHECK.
     *
     * @param token    The token id, below vocab_size.
     * @param position Its position in the sequence, below max_sequence_length.
     * @return The logits, vocab_size floats. A view into state_: valid while the Transformer
     *         lives and until the next call to forward(), which overwrites it. Copy it to keep it.
     */
    [[nodiscard]] std::span<const float> forward(std::size_t token, std::size_t position);

  private:
    MappedFile file_; /**< Declared first: built first, destroyed last. weights_ looks into it. */
    Config config_;   /**< Copy of the header, parsed from file_. */
    Weights weights_; /**< Views into file_; owns no data. */
    RunState state_;  /**< Owns every buffer the forward pass writes. */

    /**
     * @brief Where the key of `layer` at `position` starts in key_cache (and its value in
     * value_cache).
     *
     * The caches are flat: `(layer * max_sequence_length + position) * kv_head_count *
     * head_size`. From that index on come the `kv_head_count * head_size` floats of every
     * KV head of that token in that layer, one after another. Private: only forward() needs
     * to know the layout, so it is tested through forward() at two positions.
     *
     * An out-of-range layer or position is a bug in the caller and aborts with TINY_LLM_CHECK.
     *
     * @param layer    The layer, from 0 to layer_count − 1.
     * @param position The position of the token, from 0 to max_sequence_length − 1.
     * @return The index of the first float, in floats from the start of the cache.
     */
    [[nodiscard]] std::size_t cache_offset(std::size_t layer, std::size_t position) const noexcept;
};

} // namespace tiny_llm
