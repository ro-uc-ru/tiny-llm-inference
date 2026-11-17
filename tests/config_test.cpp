#include <tiny_llm/config.hpp>
#include <tiny_llm/load_error.hpp>

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <span>

namespace {
constexpr std::array<std::int32_t, 7> stories110M_header{768, 2048, 12, 12, 12, 32000, 1024};
}

TEST(ConfigTest, ParsesStories110MHeader) {
    const auto header_bytes = std::as_bytes(std::span{stories110M_header});
    const tiny_llm::Config config = tiny_llm::parse_config(header_bytes);

    EXPECT_EQ(config.embedding_size, 768u);
    EXPECT_EQ(config.feed_forward_size, 2048u);
    EXPECT_EQ(config.layer_count, 12u);
    EXPECT_EQ(config.head_count, 12u);
    EXPECT_EQ(config.kv_head_count, 12u);
    EXPECT_EQ(config.vocab_size, 32000u);
    EXPECT_EQ(config.max_sequence_length, 1024u);
    EXPECT_TRUE(config.shares_classifier_weights);
}
TEST(ConfigTest, NegativeVocabSizeMeansSeparateClassifier) {
    auto header = stories110M_header; // a modifiable copy; the original stays intact
    header[5] = -32000;               // offset 20: the sign is the shared-classifier flag

    const tiny_llm::Config config = tiny_llm::parse_config(std::as_bytes(std::span{header}));

    EXPECT_EQ(config.vocab_size, 32000u);
    EXPECT_FALSE(config.shares_classifier_weights);
}

TEST(ConfigTest, RejectsTruncatedHeader) {
    const auto header_bytes = std::as_bytes(std::span{stories110M_header});

    EXPECT_THROW(static_cast<void>(tiny_llm::parse_config(header_bytes.first(10))),
                 tiny_llm::LoadError);
}

TEST(ConfigTest, RejectsZeroHeadCount) {
    auto header = stories110M_header;
    header[3] = 0;

    EXPECT_THROW(static_cast<void>(tiny_llm::parse_config(std::as_bytes(std::span{header}))),
                 tiny_llm::LoadError);
}

TEST(ConfigTest, RejectsZeroVocabSize) {
    auto header = stories110M_header;
    header[5] = 0;

    EXPECT_THROW(static_cast<void>(tiny_llm::parse_config(std::as_bytes(std::span{header}))),
                 tiny_llm::LoadError);
}

TEST(ConfigTest, RejectsEmbeddingNotDivisibleByHeads) {
    auto header = stories110M_header;
    header[3] = 10; // 768 / 10 is not exact
    header[4] = 10; // keep head_count % kv_head_count valid, so only one check can fail

    EXPECT_THROW(static_cast<void>(tiny_llm::parse_config(std::as_bytes(std::span{header}))),
                 tiny_llm::LoadError);
}

TEST(ConfigTest, RejectsHeadsNotDivisibleByKvHeads) {
    auto header = stories110M_header;
    header[4] = 5; // 12 query heads cannot be split evenly across 5 K/V heads

    EXPECT_THROW(static_cast<void>(tiny_llm::parse_config(std::as_bytes(std::span{header}))),
                 tiny_llm::LoadError);
}

TEST(ConfigTest, ConfigIsSixtyFourBytes) {
    // 7 size_t (56 bytes) + 1 bool, padded to a multiple of 8.
    EXPECT_EQ(sizeof(tiny_llm::Config), 64u);
}
