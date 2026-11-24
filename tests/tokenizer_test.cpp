#include <tiny_llm/load_error.hpp>
#include <tiny_llm/tokenizer.hpp>

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <format>
#include <fstream>
#include <ios>
#include <string>
#include <string_view>
#include <vector>

namespace {

/** @brief One entry of a toy tokenizer.bin: its piece and its merge score. */
struct ToyToken {
    std::string piece;
    float score;
};

/**
 * @brief The toy vocabulary of the guide's "hi" example, with the real file's layout.
 *
 * Tokens 0–2 are <unk>, begin and end; 3–258 are the byte tokens <0x00>…<0xFF>; then " ", "h",
 * "i" and three merges whose scores make " h" (-1) beat "hi" (-2), as in tokenizer.bin
 * (-39 and -2659). 265 tokens in all.
 */
std::vector<ToyToken> toy_vocabulary() {
    std::vector<ToyToken> tokens{{"<unk>", 0.0f}, {"\n<s>\n", 0.0f}, {"\n</s>\n", 0.0f}};
    for (int byte_value = 0; byte_value < 256; ++byte_value) {
        tokens.push_back({std::format("<0x{:02X}>", byte_value), 0.0f});
    }
    tokens.push_back({" ", -100.0f}); // 259
    tokens.push_back({"h", -100.0f}); // 260
    tokens.push_back({"i", -100.0f}); // 261
    tokens.push_back({" h", -1.0f});  // 262
    tokens.push_back({"hi", -2.0f});  // 263
    tokens.push_back({" hi", -3.0f}); // 264
    return tokens;
}

constexpr std::size_t toy_vocab_size = 265;

/**
 * @brief Writes a toy tokenizer.bin to the temporary directory and removes it afterwards.
 *
 * The file is named after the test, so tests running in parallel never share one.
 */
class TokenizerToyTest : public ::testing::Test {
  protected:
    void SetUp() override {
        const std::string test_name =
            ::testing::UnitTest::GetInstance()->current_test_info()->name();
        toy_path_ =
            std::filesystem::temp_directory_path() / ("tiny_llm_tokenizer_" + test_name + ".bin");
    }

    void TearDown() override { std::filesystem::remove(toy_path_); }

    /**
     * @brief Writes max_token_length and then score, length and bytes of every token.
     * @param tokens The tokens to write, in order.
     */
    void write_toy_file(const std::vector<ToyToken>& tokens) const {
        std::ofstream output(toy_path_, std::ios::binary);
        const std::int32_t max_token_length = 6;
        output.write(reinterpret_cast<const char*>(&max_token_length), sizeof(max_token_length));
        for (const ToyToken& token : tokens) {
            const auto length = static_cast<std::int32_t>(token.piece.size());
            output.write(reinterpret_cast<const char*>(&token.score), sizeof(token.score));
            output.write(reinterpret_cast<const char*>(&length), sizeof(length));
            output.write(token.piece.data(), static_cast<std::streamsize>(token.piece.size()));
        }
    }

    std::filesystem::path toy_path_;
};

} // namespace

/** @brief "hi": " h" (-1) is merged before "hi" (-2), then " hi". The guide's example. */
TEST_F(TokenizerToyTest, EncodeMergesTheBestPairFirst) {
    write_toy_file(toy_vocabulary());
    const tiny_llm::Tokenizer tokenizer{toy_path_, toy_vocab_size};

    EXPECT_EQ(tokenizer.encode("hi", true), (std::vector<std::size_t>{1, 264}));
    EXPECT_EQ(tokenizer.encode("hi", false), (std::vector<std::size_t>{264}));
}

/** @brief An empty text gets no dummy prefix: only the begin token, or nothing. */
TEST_F(TokenizerToyTest, EncodeOfEmptyTextIsOnlyTheBeginToken) {
    write_toy_file(toy_vocabulary());
    const tiny_llm::Tokenizer tokenizer{toy_path_, toy_vocab_size};

    EXPECT_EQ(tokenizer.encode("", true), (std::vector<std::size_t>{1}));
    EXPECT_TRUE(tokenizer.encode("", false).empty());
}

/** @brief "ó" (C3 B3) is not in the toy vocabulary: one byte token each, byte + 3. */
TEST_F(TokenizerToyTest, EncodeFallsBackToBytes) {
    write_toy_file(toy_vocabulary());
    const tiny_llm::Tokenizer tokenizer{toy_path_, toy_vocab_size};

    EXPECT_EQ(tokenizer.encode("ó", true), (std::vector<std::size_t>{1, 259, 0xC3 + 3, 0xB3 + 3}));
}

/** @brief The space is dropped only after the begin token; byte tokens give their byte. */
TEST_F(TokenizerToyTest, DecodeRules) {
    write_toy_file(toy_vocabulary());
    const tiny_llm::Tokenizer tokenizer{toy_path_, toy_vocab_size};

    EXPECT_EQ(tokenizer.decode(1, 264), "hi");
    EXPECT_EQ(tokenizer.decode(260, 264), " hi");
    EXPECT_EQ(tokenizer.decode(1, 13), "\n"); // <0x0A>
    const std::string_view byte_zero = tokenizer.decode(1, 3);
    ASSERT_EQ(byte_zero.size(), 1u);
    EXPECT_EQ(byte_zero[0], '\0');
}

/** @brief A token outside the vocabulary is a bug in the caller: the check aborts. */
TEST_F(TokenizerToyTest, DecodeOutOfRangeAborts) {
    write_toy_file(toy_vocabulary());
    const tiny_llm::Tokenizer tokenizer{toy_path_, toy_vocab_size};

    EXPECT_DEATH(static_cast<void>(tokenizer.decode(1, toy_vocab_size)),
                 "token 265 is outside the vocabulary of 265");
}

/** @brief The config asks for more tokens than the file holds. */
TEST_F(TokenizerToyTest, ShortFileThrowsLoadError) {
    write_toy_file(toy_vocabulary());
    EXPECT_THROW(tiny_llm::Tokenizer(toy_path_, toy_vocab_size + 1), tiny_llm::LoadError);
}

/** @brief A negative length is a corrupt file, not a huge allocation. */
TEST_F(TokenizerToyTest, NegativeLengthThrowsLoadError) {
    write_toy_file(toy_vocabulary());
    {
        // Overwrite the length of token 0, right after max_token_length and its score.
        std::fstream file(toy_path_, std::ios::binary | std::ios::in | std::ios::out);
        file.seekp(8);
        const std::int32_t negative_length = -1;
        file.write(reinterpret_cast<const char*>(&negative_length), sizeof(negative_length));
    }
    EXPECT_THROW(tiny_llm::Tokenizer(toy_path_, toy_vocab_size), tiny_llm::LoadError);
}

/** @brief Without the 256 byte tokens, encode would read outside the vocabulary. */
TEST_F(TokenizerToyTest, VocabularyWithoutByteTokensThrowsLoadError) {
    write_toy_file(toy_vocabulary());
    EXPECT_THROW(tiny_llm::Tokenizer(toy_path_, 258), tiny_llm::LoadError);
}

/** @brief MappedFile throws first. */
TEST(TokenizerTest, MissingFileThrowsLoadError) {
    EXPECT_THROW(tiny_llm::Tokenizer("/nonexistent/tokenizer.bin", 32000), tiny_llm::LoadError);
}

/**
 * @brief The real tokenizer.bin: the five prompts of llama2.c's test.c (tokens from
 *        SentencePiece) and three of the guide's examples, as printed by tokens_referencia.py.
 */
TEST(TokenizerModel, EncodeMatchesTheReference) {
    const char* tokenizer_path = std::getenv("TINY_LLM_TOKENIZER_PATH");
    if (tokenizer_path == nullptr) {
        GTEST_SKIP() << "TINY_LLM_TOKENIZER_PATH is not set";
    }
    const tiny_llm::Tokenizer tokenizer{tokenizer_path, 32000};

    struct Case {
        std::string_view text;
        std::vector<std::size_t> tokens;
    };
    const std::vector<Case> cases{
        {"", {1}},
        {"I believe the meaning of life is", {1, 306, 4658, 278, 6593, 310, 2834, 338}},
        {"Simply put, the theory of relativity states that ",
         {1, 3439, 17632, 1925, 29892, 278, 6368, 310, 14215, 537, 5922, 393, 29871}},
        {"A brief message congratulating the team on the launch:\n\n        Hi everyone,\n\n"
         "        I just ",
         {1,     319, 11473, 2643, 378,  629,   271,   18099, 278, 3815, 373, 278, 6826,
          29901, 13,  13,    4706, 6324, 14332, 29892, 13,    13,  4706, 306, 925, 29871}},
        {"Translate English to French:\n\n        sea otter => loutre de mer\n        peppermint "
         "=> menthe poivrée\n        plush girafe => girafe peluche\n        cheese =>",
         {1,    4103, 9632, 4223, 304,   5176,  29901, 13,   13,   4706, 7205, 4932, 357,
          1149, 301,  449,  276,  316,   2778,  13,    4706, 1236, 407,  837,  524,  1149,
          6042, 354,  772,  440,  29878, 1318,  13,    4706, 715,  1878, 330,  3055, 1725,
          1149, 330,  3055, 1725, 4639,  28754, 13,    4706, 923,  968,  1149}},
        {"hi", {1, 7251}},
        {"🦙", {1, 29871, 243, 162, 169, 156}},
        {"canción", {1, 24174}},
    };

    for (const Case& test_case : cases) {
        EXPECT_EQ(tokenizer.encode(test_case.text, true), test_case.tokens)
            << "text: " << test_case.text;
    }
}

/** @brief The real pieces: " Once" after the begin token, " upon" after another token. */
TEST(TokenizerModel, DecodeRealPieces) {
    const char* tokenizer_path = std::getenv("TINY_LLM_TOKENIZER_PATH");
    if (tokenizer_path == nullptr) {
        GTEST_SKIP() << "TINY_LLM_TOKENIZER_PATH is not set";
    }
    const tiny_llm::Tokenizer tokenizer{tokenizer_path, 32000};

    EXPECT_EQ(tokenizer.decode(1, 9038), "Once");
    EXPECT_EQ(tokenizer.decode(9038, 2501), " upon");
    EXPECT_EQ(tokenizer.decode(1, 13), "\n");
}
