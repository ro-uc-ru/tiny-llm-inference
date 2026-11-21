#include <tiny_llm/load_error.hpp>
#include <tiny_llm/transformer.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <ios>
#include <iterator>
#include <string>
#include <type_traits>

// Rule of zero, checked by the compiler: the Transformer declares no copy or move, yet it
// moves (every member moves) and does not copy (MappedFile does not).
static_assert(!std::is_copy_constructible_v<tiny_llm::Transformer>);
static_assert(!std::is_copy_assignable_v<tiny_llm::Transformer>);
static_assert(std::is_nothrow_move_constructible_v<tiny_llm::Transformer>);

namespace {

/**
 * @brief Writes the 90-float toy model to the temporary directory and removes it afterwards.
 *
 * The file is named after the test, so tests running in parallel never share one.
 */
class TransformerToyTest : public ::testing::Test {
  protected:
    void SetUp() override {
        const std::string test_name =
            ::testing::UnitTest::GetInstance()->current_test_info()->name();
        toy_path_ = std::filesystem::temp_directory_path() / ("tiny_llm_" + test_name + ".bin");
    }

    void TearDown() override { std::filesystem::remove(toy_path_); }

    /** @brief Writes the toy header and the first @p weight_count floats 0, 1, 2… */
    void write_toy_file(std::size_t weight_count) const {
        // embedding 2, feed-forward 3, layers 2, heads 1, K/V heads 1, vocab 4, sequence 2.
        constexpr std::array<std::int32_t, 7> toy_header{2, 3, 2, 1, 1, 4, 2};
        std::array<float, 91> toy_weights{};
        for (std::size_t index = 0; index < toy_weights.size(); ++index) {
            toy_weights[index] = static_cast<float>(index);
        }

        std::ofstream output(toy_path_, std::ios::binary);
        output.write(reinterpret_cast<const char*>(toy_header.data()), sizeof(toy_header));
        output.write(reinterpret_cast<const char*>(toy_weights.data()),
                     static_cast<std::streamsize>(weight_count * sizeof(float)));
    }

    std::filesystem::path toy_path_;
};

} // namespace

/** @brief The four members build in order from a real file on disk. */
TEST_F(TransformerToyTest, BuildsFromTheToyFile) {
    write_toy_file(90);
    const tiny_llm::Transformer model{toy_path_};

    EXPECT_EQ(model.config().embedding_size, 2u);
    EXPECT_EQ(model.config().vocab_size, 4u);
}

/** @brief map_weights throws halfway through the constructor; file_ and config_ clean up. */
TEST_F(TransformerToyTest, OneFloatShortThrowsLoadError) {
    write_toy_file(89);
    EXPECT_THROW(tiny_llm::Transformer{toy_path_}, tiny_llm::LoadError);
}

/** @brief MappedFile throws first; nothing else is built. */
TEST(TransformerTest, MissingFileThrowsLoadError) {
    EXPECT_THROW(tiny_llm::Transformer{"/nonexistent/tiny_llm_model.bin"}, tiny_llm::LoadError);
}

/** @brief Skipped unless TINY_LLM_MODEL_PATH points to stories110M.bin. */
TEST(TransformerModel, Stories110MLoads) {
    const char* model_path = std::getenv("TINY_LLM_MODEL_PATH");
    if (model_path == nullptr) {
        GTEST_SKIP() << "TINY_LLM_MODEL_PATH is not set";
    }
    const tiny_llm::Transformer model{model_path};

    EXPECT_EQ(model.config().embedding_size, 768u);
    EXPECT_EQ(model.config().layer_count, 12u);
    EXPECT_EQ(model.config().vocab_size, 32000u);
}

/**
 * @brief Two forward calls in a row with stories110M, against the NumPy forward of phase 1.
 *
 * Token 1 (BOS) at position 0, then token 9038 ("Once") at position 1. The reference values
 * come from fase-2-motor-cpp/bloque-2/logits_referencia.py in the learning repo. The second
 * call reads the cache the first one left: with a wrong cache_offset, position 0 still
 * passes (it writes and reads with the same wrong formula) and position 1 fails.
 *
 * The tolerance is 1e-4: NumPy adds the products in another order, and the largest error
 * measured is 1e-5. A real bug gives errors of 0.1 or more.
 */
TEST(TransformerModel, Stories110MForward) {
    const char* model_path = std::getenv("TINY_LLM_MODEL_PATH");
    if (model_path == nullptr) {
        GTEST_SKIP() << "TINY_LLM_MODEL_PATH is not set";
    }
    tiny_llm::Transformer model{model_path};
    constexpr float logits_tolerance = 1e-4F;

    const std::array<float, 5> expected_at_position_0{-8.747768F, 1.456615F, -8.747845F, -8.747747F,
                                                      -8.747827F};
    const std::span<const float> logits_at_position_0 = model.forward(1, 0);
    ASSERT_EQ(logits_at_position_0.size(), 32000u);
    for (std::size_t index = 0; index < expected_at_position_0.size(); index++) {
        EXPECT_NEAR(logits_at_position_0[index], expected_at_position_0[index], logits_tolerance)
            << "position 0, logit " << index;
    }
    EXPECT_EQ(
        std::distance(logits_at_position_0.begin(),
                      std::max_element(logits_at_position_0.begin(), logits_at_position_0.end())),
        9038);

    // From here on, logits_at_position_0 must not be read: the next call overwrites it.
    const std::array<float, 5> expected_at_position_1{-10.438682F, -0.173467F, -10.438671F,
                                                      -10.438544F, -10.438519F};
    const std::span<const float> logits_at_position_1 = model.forward(9038, 1);
    for (std::size_t index = 0; index < expected_at_position_1.size(); index++) {
        EXPECT_NEAR(logits_at_position_1[index], expected_at_position_1[index], logits_tolerance)
            << "position 1, logit " << index;
    }
    EXPECT_EQ(
        std::distance(logits_at_position_1.begin(),
                      std::max_element(logits_at_position_1.begin(), logits_at_position_1.end())),
        2501);
}

/** @brief Skipped unless TINY_LLM_TINYLLAMA_PATH points to tinyllama.bin (GQA, own classifier). */
TEST(TransformerModel, TinyLlamaLoads) {
    const char* model_path = std::getenv("TINY_LLM_TINYLLAMA_PATH");
    if (model_path == nullptr) {
        GTEST_SKIP() << "TINY_LLM_TINYLLAMA_PATH is not set";
    }
    const tiny_llm::Transformer model{model_path};

    EXPECT_EQ(model.config().embedding_size, 2048u);
    EXPECT_EQ(model.config().kv_head_count, 4u);
    EXPECT_FALSE(model.config().shares_classifier_weights);
}
