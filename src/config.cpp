#include <tiny_llm/config.hpp>
#include <tiny_llm/load_error.hpp>

#include <bit>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <format>
#include <limits>

namespace tiny_llm {
namespace {

static_assert(std::endian::native == std::endian::little,
              "the llama2.c checkpoint format is little-endian");

constexpr std::size_t header_field_count = 7;
constexpr std::size_t header_size_in_bytes = header_field_count * sizeof(std::int32_t); // 28

/**
 * @brief Reads the little-endian int32 stored at byte offset `offset`.
 *
 * The caller guarantees that offset + 4 <= bytes.size().
 *
 * @param bytes The header bytes.
 * @param offset Where the field starts, in bytes.
 * @return The raw field value.
 */
std::int32_t read_int32(std::span<const std::byte> bytes, std::size_t offset) {
    std::int32_t raw_value;
    std::memcpy(&raw_value, bytes.data() + offset, sizeof raw_value);
    return raw_value;
}

/**
 * @brief Turns a header field that must be strictly positive into a size.
 *
 * Checks first, converts after: a negative int32 cast to size_t becomes a huge number.
 *
 * @param raw_value The field as stored in the file.
 * @param field_name The field's name, for the error message.
 * @return The value as a size.
 * @throws LoadError If the value is zero or negative.
 */
std::size_t to_positive_size(std::int32_t raw_value, const char* field_name) {
    if (raw_value <= 0) {
        throw LoadError(std::format("{} must be positive, got {}", field_name, raw_value));
    }
    return static_cast<std::size_t>(raw_value);
}

} // namespace

Config parse_config(std::span<const std::byte> bytes) {
    if (bytes.size() < header_size_in_bytes) {
        throw LoadError(std::format("a llama2.c checkpoint header needs {} bytes, got {}",
                                    header_size_in_bytes, bytes.size()));
    }

    Config config;
    config.embedding_size = to_positive_size(read_int32(bytes, 0), "embedding_size");
    config.feed_forward_size = to_positive_size(read_int32(bytes, 4), "feed_forward_size");
    config.layer_count = to_positive_size(read_int32(bytes, 8), "layer_count");
    config.head_count = to_positive_size(read_int32(bytes, 12), "head_count");
    config.kv_head_count = to_positive_size(read_int32(bytes, 16), "kv_head_count");
    // check if embedding size is % 0 with head count
    if (config.embedding_size % config.head_count != 0) {
        throw LoadError(std::format("embedding size {} is not divisible by head count {}",
                                    config.embedding_size, config.head_count));
    }
    // each K/V head is shared by head_count / kv_head_count query heads (GQA
    if (config.head_count % config.kv_head_count != 0) {
        throw LoadError(std::format("head count of {} is not divisible by kv head count of {}",
                                    config.head_count, config.kv_head_count));
    }
    const std::int32_t vocab_size_raw = read_int32(bytes, 20);
    if (vocab_size_raw == 0 || vocab_size_raw == std::numeric_limits<std::int32_t>::min()) {
        throw LoadError(std::format("vocab size {} is not a valid size", vocab_size_raw));
    }
    config.vocab_size = static_cast<std::size_t>(std::abs(vocab_size_raw));
    config.max_sequence_length = to_positive_size(read_int32(bytes, 24), "max_sequence_length");
    config.shares_classifier_weights = (vocab_size_raw > 0);
    return config;
}

} // namespace tiny_llm
