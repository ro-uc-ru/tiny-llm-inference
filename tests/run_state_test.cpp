#include <tiny_llm/config.hpp>
#include <tiny_llm/run_state.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

namespace {

/** Builds a Config the same way the engine does: from the 7 header fields. */
tiny_llm::Config config_from(const std::array<std::int32_t, 7>& header) {
    return tiny_llm::parse_config(std::as_bytes(std::span{header}));
}

/** Toy model: embedding 2, feed-forward 3, layers 2, heads 1, K/V heads 1, vocab 4, sequence 2. */
constexpr std::array<std::int32_t, 7> toy_header{2, 3, 2, 1, 1, 4, 2};
/** stories110M: no GQA. */
constexpr std::array<std::int32_t, 7> stories110M_header{768, 2048, 12, 12, 12, 32000, 1024};
/** TinyLlama 1.1B: GQA, 4 K/V heads for 32 query heads (negative vocab: own classifier). */
constexpr std::array<std::int32_t, 7> tinyllama_header{2048, 5632, 22, 32, 4, -32000, 2048};

/** @return The floats of all 11 buffers together. */
std::size_t total_floats(const tiny_llm::RunState& state) {
    return state.hidden.size() + state.normalized.size() + state.query.size() +
           state.attention_result.size() + state.projection.size() +
           state.feed_forward_hidden.size() + state.feed_forward_gate.size() +
           state.attention_scores.size() + state.logits.size() + state.key_cache.size() +
           state.value_cache.size();
}

} // namespace

/** @brief Every buffer of the toy model, by hand: 5 * 2 + 2 * 3 + 1 * 2 + 4 + 2 * 8 = 38. */
TEST(RunState, ToyModelBufferSizes) {
    const tiny_llm::RunState state = tiny_llm::make_run_state(config_from(toy_header));

    EXPECT_EQ(state.hidden.size(), 2u);
    EXPECT_EQ(state.normalized.size(), 2u);
    EXPECT_EQ(state.query.size(), 2u);
    EXPECT_EQ(state.attention_result.size(), 2u);
    EXPECT_EQ(state.projection.size(), 2u);
    EXPECT_EQ(state.feed_forward_hidden.size(), 3u);
    EXPECT_EQ(state.feed_forward_gate.size(), 3u);
    EXPECT_EQ(state.attention_scores.size(), 2u); // head_count 1 * max_sequence_length 2
    EXPECT_EQ(state.logits.size(), 4u);
    EXPECT_EQ(state.key_cache.size(), 8u); // layers 2 * sequence 2 * K/V heads 1 * head_size 2
    EXPECT_EQ(state.value_cache.size(), 8u);
    EXPECT_EQ(total_floats(state), 38u);
}

/** @brief The buffers start at zero, like calloc or np.zeros. */
TEST(RunState, ToyModelStartsAtZero) {
    const tiny_llm::RunState state = tiny_llm::make_run_state(config_from(toy_header));
    const auto is_zero = [](float value) { return value == 0.0F; };

    EXPECT_TRUE(std::ranges::all_of(state.hidden, is_zero));
    EXPECT_TRUE(std::ranges::all_of(state.logits, is_zero));
    EXPECT_TRUE(std::ranges::all_of(state.key_cache, is_zero));
    EXPECT_TRUE(std::ranges::all_of(state.value_cache, is_zero));
}

/** @brief 18'926'592 floats, 75.7 MB; the two caches are 2 * 12 * 1024 * 768 of them. */
TEST(RunState, Stories110MTotal) {
    const tiny_llm::RunState state = tiny_llm::make_run_state(config_from(stories110M_header));

    EXPECT_EQ(state.key_cache.size(), 9'437'184u);
    EXPECT_EQ(total_floats(state), 18'926'592u);
}

/** @brief GQA: each cache position holds 4 * 64 = 256 floats, not 2048. */
TEST(RunState, TinyLlamaCacheWithGroupedQueryAttention) {
    const tiny_llm::RunState state = tiny_llm::make_run_state(config_from(tinyllama_header));

    EXPECT_EQ(state.key_cache.size(), 22u * 2048u * 256u);
    EXPECT_EQ(state.attention_scores.size(), 32u * 2048u); // query heads, not K/V heads
    EXPECT_EQ(total_floats(state), 23'187'712u);
}
