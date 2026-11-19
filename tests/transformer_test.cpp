#include <tiny_llm/load_error.hpp>
#include <tiny_llm/transformer.hpp>

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <ios>
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
