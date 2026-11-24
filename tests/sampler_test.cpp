#include <tiny_llm/sampler.hpp>

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

// Only sample() is public (decision 4 of the guide), so every test goes through it with seed
// 42, whose first three coins are 0.339085, 0.782256 and 0.790137. The expected tokens were
// computed in Python with float32.

namespace {

constexpr std::uint64_t seed = 42;

/**
 * @brief 100 equal logits: every token has probability 0.01, so the chosen token is
 *        floor(coin · 100), which checks each coin of the xorshift to two decimals.
 */
TEST(Sampler, UniformLogitsRevealTheCoins) {
    tiny_llm::Sampler sampler{100, 1.0f, 0.0f, seed};
    const std::array<std::size_t, 3> expected_tokens{33, 78, 79};

    for (const std::size_t expected_token : expected_tokens) {
        std::vector<float> logits(100, 0.0f);
        EXPECT_EQ(sampler.sample(logits), expected_token);
    }
}

/**
 * @brief Top-p 0.9 over probabilities in increasing order, {0.05, 0.15, 0.3, 0.5}.
 *
 * The nucleus is tokens 3, 2, 1 (0.5 + 0.3 + 0.15 = 0.95 > 0.9). Scaled coins 0.322, 0.743 and
 * 0.751 give tokens 3, 2, 2. Catches a missing sort, returning the position after sorting
 * instead of the token, and token 0 (outside the nucleus) coming out.
 */
TEST(Sampler, TopPSortsAndCutsTheNucleus) {
    tiny_llm::Sampler sampler{4, 1.0f, 0.9f, seed};
    const std::array<std::size_t, 3> expected_tokens{3, 2, 2};

    for (const std::size_t expected_token : expected_tokens) {
        // log of the probabilities: with temperature 1, softmax gives them back.
        std::vector<float> logits{std::log(0.05f), std::log(0.15f), std::log(0.3f), std::log(0.5f)};
        EXPECT_EQ(sampler.sample(logits), expected_token);
    }
}

/** @brief Temperature 0 is the argmax, and the logits are left untouched. */
TEST(Sampler, TemperatureZeroIsArgmax) {
    tiny_llm::Sampler sampler{3, 0.0f, 0.0f, seed};
    std::vector<float> logits{1.0f, 5.0f, 2.0f};

    EXPECT_EQ(sampler.sample(logits), 1u);
    EXPECT_EQ(logits, (std::vector<float>{1.0f, 5.0f, 2.0f}));
}

/** @brief Seed 0 would make every coin 0: a bug in the caller, so the check aborts. */
TEST(SamplerDeathTest, ZeroSeedAborts) {
    EXPECT_DEATH(tiny_llm::Sampler(32000, 1.0f, 0.0f, 0), "seed must not be 0");
}

/** @brief Empty logits have no token to choose: the check aborts. */
TEST(SamplerDeathTest, EmptyLogitsAbort) {
    tiny_llm::Sampler sampler{1, 1.0f, 0.0f, seed};
    std::vector<float> logits;
    EXPECT_DEATH(static_cast<void>(sampler.sample(logits)), "logits must not be empty");
}

} // namespace
