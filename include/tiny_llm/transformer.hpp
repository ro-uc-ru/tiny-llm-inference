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
     * @brief Runs one token through the model. Declared here, written in step 4.
     *
     * @param token    The token id, below vocab_size.
     * @param position Its position in the sequence, below max_sequence_length.
     * @return The logits, a view into state_ that the next call overwrites.
     */
    [[nodiscard]] std::span<const float> forward(std::size_t token, std::size_t position);

  private:
    MappedFile file_; /**< Declared first: built first, destroyed last. weights_ looks into it. */
    Config config_;   /**< Copy of the header, parsed from file_. */
    Weights weights_; /**< Views into file_; owns no data. */
    RunState state_;  /**< Owns every buffer the forward pass writes. */
};

} // namespace tiny_llm
