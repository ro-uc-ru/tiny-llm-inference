#include <tiny_llm/check.hpp>
#include <tiny_llm/load_error.hpp>
#include <tiny_llm/mapped_file.hpp>
#include <tiny_llm/tokenizer.hpp>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <format>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace tiny_llm {

namespace {

/**
 * @brief Whether `byte` continues a UTF-8 character instead of starting one.
 *
 * Continuation bytes start with the bits 10: `(byte & 0xC0) == 0x80`. In "ó" = C3 B3, B3 is
 * one. The cast to `unsigned char` comes first because `char` is signed on Apple Silicon, and
 * 0xB3 would be the negative number -77.
 *
 * @param byte One byte of the text.
 * @return True if it continues the previous character.
 */
bool is_continuation_byte(char byte) {
    return (static_cast<unsigned char>(byte) & 0xC0) == 0x80;
}

} // namespace

/**
 * @brief Walks tokenizer.bin with a cursor, copying every piece and score out of the mapping.
 *
 * The MappedFile is a local: the file is unmapped when the constructor returns, which is why
 * every piece is copied into its own std::string.
 *
 * @param path       The tokenizer.bin file.
 * @param vocab_size The number of tokens to read.
 * @throws LoadError If the vocabulary has no room for the 256 byte tokens, or the file is
 *                   missing, ends early or holds a negative length.
 */
Tokenizer::Tokenizer(const std::filesystem::path& path, std::size_t vocab_size) {
    // encode() falls back to byte tokens 3…258 and then reads their pieces: a smaller
    // vocabulary would read outside pieces_. vocab_size comes from the model file, so a wrong
    // one is an environment error, not a bug here.
    if (vocab_size < first_byte_token + byte_values_.size()) {
        throw LoadError(std::format("a vocabulary of {} tokens has no room for the 256 byte tokens",
                                    vocab_size));
    }
    const MappedFile file{path};
    const std::span<const std::byte> bytes = file.bytes();

    pieces_.reserve(vocab_size);
    scores_.reserve(vocab_size);
    piece_to_token_.reserve(vocab_size);

    // The three lambdas share the cursor by reference. `require` compares against
    // `bytes.size() - cursor`, which cannot wrap because the cursor never passes the end.
    // The readers copy with std::memcpy, not reinterpret_cast: after the first piece, the
    // floats are not aligned to 4 bytes.
    std::size_t cursor = 0;
    auto require = [&](std::size_t byte_count) {
        if (byte_count > bytes.size() - cursor) {
            throw LoadError(std::format("tokenizer.bin has {} bytes, but byte {} needs {} more",
                                        bytes.size(), cursor, byte_count));
        }
    };
    auto read_int32 = [&]() -> std::int32_t {
        std::int32_t value = 0;
        require(sizeof(value));
        std::memcpy(&value, bytes.data() + cursor, sizeof(value));
        cursor += sizeof(value);
        return value;
    };
    auto read_float = [&]() -> float {
        float value = 0.0f;
        require(sizeof(value));
        std::memcpy(&value, bytes.data() + cursor, sizeof(value));
        cursor += sizeof(value);
        return value;
    };

    read_int32(); // max_token_length: not used, but it is in the file

    for (std::size_t token = 0; token < vocab_size; ++token) {
        const float score = read_float();
        const std::int32_t length = read_int32();
        if (length < 0) {
            throw LoadError(std::format("tokenizer.bin: token {} has a negative length, {}",
                                        token, length));
        }
        const auto byte_count = static_cast<std::size_t>(length);
        require(byte_count);
        pieces_.push_back(
            std::string(reinterpret_cast<const char*>(bytes.data() + cursor), byte_count));
        scores_.push_back(score);
        cursor += byte_count;
    }

    // After the loop, when pieces_ no longer grows. The map copies each key anyway, so it does
    // not depend on pieces_ (decision 2 of the guide).
    for (std::size_t token = 0; token < pieces_.size(); ++token) {
        piece_to_token_.emplace(pieces_[token], token);
    }

    for (std::size_t value = 0; value < byte_values_.size(); ++value) {
        byte_values_[value] = static_cast<char>(value);
    }
}

/**
 * @brief BPE in three parts: split into characters (or bytes), merge the best pair until none
 *        is left, and put the begin token first.
 *
 * Quadratic in the number of characters, like run.c: fine for prompts of tens of tokens.
 *
 * @param text            UTF-8 text.
 * @param add_begin_token Whether to put begin_token first.
 * @return The tokens of `text`.
 */
std::vector<std::size_t> Tokenizer::encode(std::string_view text, bool add_begin_token) const {
    std::vector<std::size_t> tokens;
    if (add_begin_token) {
        tokens.push_back(begin_token);
    }
    if (text.empty()) {
        return tokens; // no dummy prefix for an empty text
    }

    // SentencePiece's dummy prefix: "Once" is encoded as " Once".
    const std::string spaced_text = " " + std::string{text};

    // Part 1: one token per UTF-8 character, a first byte plus the continuation bytes after
    // it; a character that is not in the vocabulary becomes one byte token per byte.
    // substr(start, count) is Python's text[start:start + count].
    std::vector<std::size_t> merging;
    std::size_t character_start = 0;
    while (character_start < spaced_text.size()) {
        std::size_t character_end = character_start + 1;
        while (character_end < spaced_text.size() &&
               is_continuation_byte(spaced_text[character_end])) {
            ++character_end;
        }
        const std::string character =
            spaced_text.substr(character_start, character_end - character_start);
        const auto found = piece_to_token_.find(character);
        if (found != piece_to_token_.end()) {
            merging.push_back(found->second);
        } else {
            for (const char byte : character) {
                merging.push_back(std::size_t{static_cast<unsigned char>(byte)} + first_byte_token);
            }
        }
        character_start = character_end;
    }

    // Part 2: each round merges the neighbouring pair whose concatenation is the token with the
    // highest score; `>` and not `>=`, so ties go to the leftmost pair. `index + 1 < size`
    // instead of `index < size - 1`: no subtraction that could wrap around.
    while (true) {
        bool found_pair = false;
        std::size_t best_index = 0;
        std::size_t best_token = 0;
        float best_score = 0.0f;
        for (std::size_t index = 0; index + 1 < merging.size(); ++index) {
            const std::string pair_text = pieces_[merging[index]] + pieces_[merging[index + 1]];
            const auto found = piece_to_token_.find(pair_text);
            if (found != piece_to_token_.end() &&
                (!found_pair || scores_[found->second] > best_score)) {
                found_pair = true;
                best_index = index;
                best_token = found->second;
                best_score = scores_[found->second];
            }
        }
        if (!found_pair) {
            break;
        }
        // The pair becomes one token: shift the rest one place left and drop the last one.
        merging[best_index] = best_token;
        for (std::size_t index = best_index + 1; index + 1 < merging.size(); ++index) {
            merging[index] = merging[index + 1];
        }
        merging.pop_back();
    }

    // Part 3: the merged tokens go after the begin token.
    for (const std::size_t token : merging) {
        tokens.push_back(token);
    }
    return tokens;
}

/**
 * @brief Byte tokens give their byte; other pieces lose their leading space after the begin
 *        token.
 *
 * Every view points into this tokenizer (byte_values_ or pieces_), never into a temporary.
 *
 * @param previous_token The token printed just before.
 * @param token          The token to turn into text.
 * @return A view valid while the tokenizer lives.
 */
std::string_view Tokenizer::decode(std::size_t previous_token, std::size_t token) const {
    TINY_LLM_CHECK(token < pieces_.size(), std::format("token {} is outside the vocabulary of {}",
                                                       token, pieces_.size()));

    if (token >= first_byte_token && token < first_byte_token + byte_values_.size()) {
        return std::string_view{&byte_values_[token - first_byte_token], 1};
    }

    // Not const: remove_prefix moves the view one character forward; the piece is untouched.
    std::string_view piece = pieces_[token];
    if (previous_token == begin_token && piece.starts_with(' ')) {
        piece.remove_prefix(1);
    }
    return piece;
}

} // namespace tiny_llm
