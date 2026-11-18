#include <tiny_llm/config.hpp>
#include <tiny_llm/load_error.hpp>
#include <tiny_llm/mapped_file.hpp>
#include <tiny_llm/weights.hpp>

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <span>
#include <vector>

namespace {

/**
 * @brief Parses a header given as its seven int32 fields.
 * @param header The fields in file order, as llama2.c writes them.
 * @return The validated config.
 */
tiny_llm::Config config_from(const std::array<std::int32_t, 7>& header) {
    return tiny_llm::parse_config(std::as_bytes(std::span{header}));
}

/** Toy model: 90 floats, small enough to count by hand. Shared classifier. */
constexpr std::array<std::int32_t, 7> toy_header{2, 3, 2, 1, 1, 4, 2};
/** stories110M: no GQA, shared classifier. */
constexpr std::array<std::int32_t, 7> stories110M_header{768, 2048, 12, 12, 12, 32000, 1024};
/** TinyLlama 1.1B: GQA (4 K/V heads for 32 query heads), own classifier (negative vocab). */
constexpr std::array<std::int32_t, 7> tinyllama_header{2048, 5632, 22, 32, 4, -32000, 2048};

/**
 * @brief A whole checkpoint in memory: the header, then float k at float offset k.
 *
 * memcpy, not a cast: writing a float through bytes that are not a float is undefined
 * behaviour; copying its representation is not.
 *
 * @param header      The seven header fields.
 * @param float_count How many floats to write after the header.
 * @return The file bytes, as map_weights expects them.
 */
std::vector<std::byte> toy_file(const std::array<std::int32_t, 7>& header,
                                std::size_t float_count) {
    std::vector<std::byte> bytes(tiny_llm::header_size_in_bytes + sizeof(float) * float_count);
    std::memcpy(bytes.data(), header.data(), tiny_llm::header_size_in_bytes);
    for (std::size_t offset = 0; offset < float_count; ++offset) {
        const auto value = static_cast<float>(offset);
        std::memcpy(bytes.data() + tiny_llm::header_size_in_bytes + sizeof(float) * offset, &value,
                    sizeof value);
    }
    return bytes;
}

} // namespace

/** @brief The toy model, term by term: 8+4+8+8+8+8+4+12+12+12+2+4 (RoPE) + 0 (shared). */
TEST(WeightFloatCountTest, ToyModelIsNinety) {
    EXPECT_EQ(tiny_llm::weight_float_count(config_from(toy_header)), 90u);
}

/** @brief A classifier of its own adds vocab_size * embedding_size = 4 * 2 floats. */
TEST(WeightFloatCountTest, OwnClassifierAddsVocabTimesEmbedding) {
    auto header = toy_header;
    header[5] = -header[5];
    EXPECT_EQ(tiny_llm::weight_float_count(config_from(header)), 98u);
}

/** @brief 28 + 4 * 109'595'392 = 438'381'596 bytes, the size of stories110M.bin. */
TEST(WeightFloatCountTest, Stories110M) {
    EXPECT_EQ(tiny_llm::weight_float_count(config_from(stories110M_header)), 109'595'392u);
}

/**
 * @brief GQA: K and V have 4 * 64 = 256 rows, not 2048.
 *
 * Computed from the header; still to be checked against the size of the exported file.
 */
TEST(WeightFloatCountTest, TinyLlamaWithGroupedQueryAttention) {
    EXPECT_EQ(tiny_llm::weight_float_count(config_from(tinyllama_header)), 1'100'179'456u);
}

/** @brief One float short and one float too many are both the file's fault: LoadError. */
TEST(MapWeightsTest, RejectsWrongFileSize) {
    const tiny_llm::Config config = config_from(toy_header);
    EXPECT_THROW(static_cast<void>(tiny_llm::map_weights(toy_file(toy_header, 89), config)),
                 tiny_llm::LoadError);
    EXPECT_THROW(static_cast<void>(tiny_llm::map_weights(toy_file(toy_header, 91), config)),
                 tiny_llm::LoadError);
}

/** @brief Every float holds its own offset, so each value says where the view starts. */
TEST(MapWeightsTest, ToyModelViewsTheRightFloats) {
    const std::vector<std::byte> bytes = toy_file(toy_header, 90);
    const tiny_llm::Weights weights = tiny_llm::map_weights(bytes, config_from(toy_header));

    ASSERT_EQ(weights.query_projection.size(), 2u);
    EXPECT_EQ(weights.query_projection[1].at(0, 0), 16.0f); // 12 + one layer of 2 x 2

    const tiny_llm::Tensor& output_layer_0 = weights.feed_forward_output[0];
    EXPECT_EQ(output_layer_0.rows(), 2u);
    EXPECT_EQ(output_layer_0.columns(), 3u);
    EXPECT_EQ(output_layer_0.at(1, 0), 63.0f);
    EXPECT_EQ(output_layer_0.at(1, 2), 65.0f);

    EXPECT_EQ(weights.final_norm.at(1, 0), 85.0f);
    EXPECT_EQ(weights.classifier.at(3, 1), 7.0f);
    // Shared: the classifier looks at the embedding's floats, it does not copy them.
    EXPECT_EQ(weights.classifier.data().data(), weights.token_embedding.data().data());
}

namespace {

/**
 * @brief The "done when" of the step, for one real checkpoint: the last view ends where the
 *        file ends.
 *
 * With a shared classifier the last view is final_norm, followed by the RoPE tables; with
 * its own classifier, the classifier comes last.
 *
 * @param model_path Path to a llama2.c checkpoint.
 */
void expect_walk_ends_at_end_of_file(const char* model_path) {
    const tiny_llm::MappedFile file{model_path};
    const tiny_llm::Config config = tiny_llm::parse_config(file.bytes());
    const tiny_llm::Weights weights = tiny_llm::map_weights(file.bytes(), config);

    const std::size_t head_size = config.embedding_size / config.head_count;
    const std::size_t rope_float_count = config.max_sequence_length * head_size;
    const std::span<const float> final_norm = weights.final_norm.data();
    const std::span<const float> classifier = weights.classifier.data();
    const float* last_end = config.shares_classifier_weights
                                ? final_norm.data() + final_norm.size() + rope_float_count
                                : classifier.data() + classifier.size();
    const auto* file_end =
        reinterpret_cast<const float*>(file.bytes().data() + file.bytes().size());
    EXPECT_EQ(last_end, file_end);
    EXPECT_EQ(weights.classifier.rows(), config.vocab_size);
    EXPECT_EQ(weights.classifier.columns(), config.embedding_size);
}

} // namespace

/** @brief Skipped unless TINY_LLM_MODEL_PATH points to stories110M.bin (shared classifier). */
TEST(MapWeightsModel, Stories110MWalkEndsAtTheEndOfTheFile) {
    const char* model_path = std::getenv("TINY_LLM_MODEL_PATH");
    if (model_path == nullptr) {
        GTEST_SKIP() << "TINY_LLM_MODEL_PATH is not set";
    }
    expect_walk_ends_at_end_of_file(model_path);
}

/** @brief Skipped unless TINY_LLM_TINYLLAMA_PATH points to tinyllama.bin (GQA, own classifier). */
TEST(MapWeightsModel, TinyLlamaWalkEndsAtTheEndOfTheFile) {
    const char* model_path = std::getenv("TINY_LLM_TINYLLAMA_PATH");
    if (model_path == nullptr) {
        GTEST_SKIP() << "TINY_LLM_TINYLLAMA_PATH is not set";
    }
    expect_walk_ends_at_end_of_file(model_path);
}
