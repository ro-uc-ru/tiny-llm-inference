#include <tiny_llm/options.hpp>

#include <gtest/gtest.h>

#include <cstdint>
#include <initializer_list>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

/**
 * @brief A fake argv: owns the texts and hands out the char* that parse_options expects.
 *
 * main receives char** from the operating system; a test has to build it. The strings own the
 * characters and the pointers look into them, so the strings must not change after the
 * pointers are taken: both vectors are filled once, in the constructor.
 */
class FakeArguments {
  public:
    /**
     * @brief Copies the texts and takes a pointer to each one.
     * @param texts The arguments, program name included.
     */
    FakeArguments(std::initializer_list<std::string> texts) : texts_{texts} {
        pointers_.reserve(texts_.size());
        for (std::string& text : texts_) {
            pointers_.push_back(text.data());
        }
    }

    /** @brief The arguments as main sees them, valid while this object lives. */
    [[nodiscard]] std::span<char*> view() noexcept { return pointers_; }

  private:
    std::vector<std::string> texts_; /**< Owns the characters. */
    std::vector<char*> pointers_; /**< Look into texts_; never reallocated after the constructor. */
};

// --- parse_float ----------------------------------------------------------------------------

TEST(ParseFloat, ReadsAWholeNumber) {
    EXPECT_EQ(tiny_llm::parse_float("-t", "0"), 0.0f);
    EXPECT_EQ(tiny_llm::parse_float("-p", "0.9"), 0.9f);
    EXPECT_EQ(tiny_llm::parse_float("-t", "-1.5"), -1.5f);
}

TEST(ParseFloat, RejectsCharactersLeftOver) {
    // std::stof alone would return 0.9.
    EXPECT_THROW(static_cast<void>(tiny_llm::parse_float("-p", "0.9x")), std::invalid_argument);
}

TEST(ParseFloat, RejectsTextThatIsNotANumber) {
    EXPECT_THROW(static_cast<void>(tiny_llm::parse_float("-t", "abc")), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(tiny_llm::parse_float("-t", "")), std::invalid_argument);
}

TEST(ParseFloat, RejectsANumberTooLargeForAFloat) {
    EXPECT_THROW(static_cast<void>(tiny_llm::parse_float("-t", "1e50")), std::out_of_range);
}

// --- parse_count ----------------------------------------------------------------------------

TEST(ParseCount, ReadsAWholeInteger) {
    EXPECT_EQ(tiny_llm::parse_count("-n", "0"), 0U);
    EXPECT_EQ(tiny_llm::parse_count("-n", "256"), 256U);
    EXPECT_EQ(tiny_llm::parse_count("-s", "18446744073709551615"), UINT64_MAX);
}

TEST(ParseCount, RejectsASignOrALeadingSpace) {
    // std::stoul alone would wrap "-5" and " -5" around to 18446744073709551611.
    EXPECT_THROW(static_cast<void>(tiny_llm::parse_count("-n", "-5")), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(tiny_llm::parse_count("-n", " -5")), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(tiny_llm::parse_count("-n", "+5")), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(tiny_llm::parse_count("-n", " 5")), std::invalid_argument);
}

TEST(ParseCount, RejectsCharactersLeftOver) {
    // std::stoul alone would return 1.
    EXPECT_THROW(static_cast<void>(tiny_llm::parse_count("-n", "1e9")), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(tiny_llm::parse_count("-n", "5x")), std::invalid_argument);
}

TEST(ParseCount, RejectsAnEmptyText) {
    EXPECT_THROW(static_cast<void>(tiny_llm::parse_count("-n", "")), std::invalid_argument);
}

TEST(ParseCount, RejectsANumberTooLargeFor64Bits) {
    EXPECT_THROW(static_cast<void>(tiny_llm::parse_count("-s", "99999999999999999999")),
                 std::out_of_range);
}

TEST(ParseCount, TheMessageNamesTheFlagAndTheText) {
    try {
        static_cast<void>(tiny_llm::parse_count("-n", "-5"));
        FAIL() << "expected std::invalid_argument";
    } catch (const std::invalid_argument& error) {
        EXPECT_STREQ(error.what(), "-n expects a non-negative integer, got -5");
    }
}

// --- parse_options --------------------------------------------------------------------------

TEST(ParseOptions, OnlyTheCheckpointGivesTheDefaults) {
    FakeArguments arguments{"tiny_llm", "model.bin"};
    const tiny_llm::Options options = tiny_llm::parse_options(arguments.view());

    EXPECT_EQ(options.checkpoint_path.string(), "model.bin");
    EXPECT_EQ(options.tokenizer_path.string(), "tokenizer.bin");
    EXPECT_EQ(options.temperature, 1.0f);
    EXPECT_EQ(options.top_p, 0.9f);
    EXPECT_EQ(options.steps, 256U);
    EXPECT_EQ(options.seed, 0U);
    EXPECT_TRUE(options.prompt.empty());
}

/** @brief The toy example of the guide: every flag lands in its own field. */
TEST(ParseOptions, EveryFlagFillsItsField) {
    FakeArguments arguments{"tiny_llm", "model.bin", "-t", "0",  "-p",        "0.5", "-n",
                            "5",        "-s",        "42", "-i", "Once upon", "-z",  "other.bin"};
    const tiny_llm::Options options = tiny_llm::parse_options(arguments.view());

    EXPECT_EQ(options.checkpoint_path.string(), "model.bin");
    EXPECT_EQ(options.temperature, 0.0f);
    EXPECT_EQ(options.top_p, 0.5f);
    EXPECT_EQ(options.steps, 5U);
    EXPECT_EQ(options.seed, 42U);
    EXPECT_EQ(options.prompt, "Once upon");
    EXPECT_EQ(options.tokenizer_path.string(), "other.bin");
}

TEST(ParseOptions, AFlagGivenTwiceKeepsTheLastValue) {
    FakeArguments arguments{"tiny_llm", "model.bin", "-n", "5", "-n", "7"};
    EXPECT_EQ(tiny_llm::parse_options(arguments.view()).steps, 7U);
}

TEST(ParseOptions, AppliesTheCorrectionsOfRunC) {
    FakeArguments negative_temperature{"tiny_llm", "model.bin", "-t", "-1"};
    EXPECT_EQ(tiny_llm::parse_options(negative_temperature.view()).temperature, 0.0f);

    FakeArguments top_p_above_one{"tiny_llm", "model.bin", "-p", "1.5"};
    EXPECT_EQ(tiny_llm::parse_options(top_p_above_one.view()).top_p, 0.9f);

    FakeArguments negative_top_p{"tiny_llm", "model.bin", "-p", "-0.1"};
    EXPECT_EQ(tiny_llm::parse_options(negative_top_p.view()).top_p, 0.9f);

    // The edges of [0, 1] are kept.
    FakeArguments top_p_one{"tiny_llm", "model.bin", "-p", "1"};
    EXPECT_EQ(tiny_llm::parse_options(top_p_one.view()).top_p, 1.0f);
}

TEST(ParseOptions, RejectsAMissingCheckpoint) {
    FakeArguments arguments{"tiny_llm"};
    EXPECT_THROW(static_cast<void>(tiny_llm::parse_options(arguments.view())),
                 std::invalid_argument);
}

TEST(ParseOptions, RejectsAFlagWithoutAValue) {
    FakeArguments arguments{"tiny_llm", "model.bin", "-t"};
    EXPECT_THROW(static_cast<void>(tiny_llm::parse_options(arguments.view())),
                 std::invalid_argument);
}

TEST(ParseOptions, RejectsAnUnknownFlag) {
    // -m is run.c's mode flag, which this engine does not have.
    FakeArguments arguments{"tiny_llm", "model.bin", "-m", "chat"};
    EXPECT_THROW(static_cast<void>(tiny_llm::parse_options(arguments.view())),
                 std::invalid_argument);
}

TEST(ParseOptions, ABadNumberReachesTheCaller) {
    FakeArguments bad_count{"tiny_llm", "model.bin", "-n", "-5"};
    EXPECT_THROW(static_cast<void>(tiny_llm::parse_options(bad_count.view())),
                 std::invalid_argument);

    FakeArguments too_large{"tiny_llm", "model.bin", "-s", "99999999999999999999"};
    EXPECT_THROW(static_cast<void>(tiny_llm::parse_options(too_large.view())), std::out_of_range);
}

} // namespace
