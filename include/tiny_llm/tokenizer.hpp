#pragma once

#include <array>
#include <cstddef>
#include <filesystem>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace tiny_llm {

/** @brief The begin-of-sequence token; its piece is \<s\> between two newlines. */
inline constexpr std::size_t begin_token = 1;

/** @brief The token of byte 0, `<0x00>`; the token of byte b is b + 3, up to `<0xFF>` = 258. */
inline constexpr std::size_t first_byte_token = 3;

/**
 * @brief Text to tokens (encode, BPE) and tokens to text (decode), from llama2.c's tokenizer.bin.
 *
 * File format: an int32 `max_token_length`, then, for each of the vocab_size tokens, a float
 * score, an int32 length and that many bytes (no terminating '\0'). The number of tokens is not
 * in the file: it comes from Config::vocab_size.
 *
 * The tokenizer copies every piece into its own `std::string`, so it does not keep the file
 * mapped: the MappedFile lives only inside the constructor.
 */
class Tokenizer {
  public:
    /**
     * @brief Reads every piece and score from `path` and builds the lookup tables.
     *
     * @param path       The tokenizer.bin file.
     * @param vocab_size The number of tokens to read (Config::vocab_size).
     * @throws LoadError If vocab_size is below 259 (no room for the byte tokens), or the file
     *                   is missing, or ends before vocab_size tokens, or holds a negative length.
     */
    Tokenizer(const std::filesystem::path& path, std::size_t vocab_size);

    /**
     * @brief Byte-pair encoding of `text`.
     *
     * An empty text gives only the begin token (if asked for). Otherwise a space is prepended
     * (SentencePiece's dummy prefix), every UTF-8 character becomes its token, or one token per
     * byte if it is not in the vocabulary, and then the neighbouring pair whose concatenation is
     * the token with the best score is merged, again and again, until no pair can be merged.
     * Ties go to the leftmost pair.
     *
     * Example: "hi" → " ", "h", "i" → " h" (score -39) beats "hi" (-2659) → " hi" → [1, 7251].
     *
     * @param text            UTF-8 text.
     * @param add_begin_token Whether to put begin_token first.
     * @return The tokens of `text`.
     */
    [[nodiscard]] std::vector<std::size_t> encode(std::string_view text,
                                                  bool add_begin_token) const;

    /**
     * @brief The text of `token`, with llama2.c's two rules.
     *
     * After the begin token, a leading space is dropped (" Once" → "Once"). A byte token
     * (3…258) gives that single byte, token - first_byte_token. Any other token gives its piece.
     * The result may be part of a UTF-8 character: a 4-byte emoji comes in four calls.
     *
     * A token out of range is a bug in the caller and aborts with TINY_LLM_CHECK.
     *
     * @param previous_token The token printed just before; begin_token at the start.
     * @param token          The token to turn into text.
     * @return A view into this tokenizer (pieces_ or byte_values_), valid while it lives.
     */
    [[nodiscard]] std::string_view decode(std::size_t previous_token, std::size_t token) const;

  private:
    std::vector<std::string> pieces_;                             /**< Token → text. */
    std::vector<float> scores_;                                   /**< Token → merge score. */
    std::unordered_map<std::string, std::size_t> piece_to_token_; /**< Text → token. */
    std::array<char, 256> byte_values_{};                         /**< 0…255, for byte tokens. */
};

} // namespace tiny_llm
