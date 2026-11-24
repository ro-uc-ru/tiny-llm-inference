#include <tiny_llm/check.hpp>
#include <tiny_llm/kernels.hpp>
#include <tiny_llm/sampler.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <span>

namespace tiny_llm {

namespace {

/**
 * @brief Multinomial sampling: walks the cumulative probabilities until one exceeds the coin.
 *
 * With probabilities {0.1, 0.2, 0.7} the cumulative values are 0.1, 0.3, 1.0; a coin of 0.25
 * gives token 1. If rounding leaves the coin above every cumulative value (the probabilities
 * add up to 0.9999999), returns the last token.
 *
 * @param probabilities The output of softmax; sums to 1.
 * @param coin          A number in [0, 1).
 * @return The chosen token, an index into `probabilities`.
 */
std::size_t sample_multinomial(std::span<const float> probabilities, float coin) {
    float running_sum = 0.0f;
    for (std::size_t token = 0; token < probabilities.size(); token++) {
        running_sum += probabilities[token];
        if (coin < running_sum) {
            return token;
        }
    }
    // Only reached when rounding leaves the sum just below 1 and the coin above it.
    return probabilities.size() - 1;
}

} // namespace

/**
 * @brief Stores the settings and reserves the top-p candidates once.
 *
 * The members are initialised in declaration order. candidates_ is not in the list, so it
 * starts empty; reserving here means sample() never allocates.
 *
 * @param vocab_size  The number of logits sample() will receive.
 * @param temperature 0 for argmax.
 * @param top_p       The nucleus size; top-p is on only for values in (0, 1).
 * @param seed        The initial xorshift state; must not be 0.
 */
Sampler::Sampler(std::size_t vocab_size, float temperature, float top_p, std::uint64_t seed)
    : temperature_{temperature}, top_p_{top_p}, random_state_{seed} {
    TINY_LLM_CHECK(random_state_ != 0, "seed must not be 0, a xorshift of 0 stays 0 forever");
    candidates_.reserve(vocab_size);
}

/**
 * @brief Argmax at temperature 0; otherwise temperature, softmax, one coin and multinomial or
 *        top-p, in the same order as run.c so the same seed gives the same tokens.
 *
 * @param logits The logits; overwritten with probabilities when temperature > 0.
 * @return The chosen token.
 */
std::size_t Sampler::sample(std::span<float> logits) {
    TINY_LLM_CHECK(!logits.empty(), "logits must not be empty");

    // Greedy decoding draws no coin, so it leaves the generator where it was.
    if (temperature_ == 0.0f) {
        const auto largest = std::max_element(logits.begin(), logits.end());
        return static_cast<std::size_t>(std::distance(logits.begin(), largest));
    }

    for (float& logit : logits) {
        logit /= temperature_;
    }
    softmax(logits); // in place: from here on, `logits` holds probabilities

    // Exactly one coin per token, also with top-p.
    const float coin = next_random_float();
    if (top_p_ > 0.0f && top_p_ < 1.0f) {
        return sample_top_p(logits, coin);
    }
    return sample_multinomial(logits, coin);
}

/**
 * @brief Top-p: sort, cut the nucleus where the sum passes top_p, and sample inside it.
 *
 * Example with {0.5, 0.3, 0.15, 0.05}, top_p 0.9 and coin 0.9: the sums are 0.5, 0.8, 0.95, so
 * the nucleus is the first three tokens; the scaled coin is 0.9 · 0.95 = 0.855, which falls in
 * the third one.
 *
 * run.c first drops the tokens below (1 - top_p) / (n - 1) to sort fewer; it does not change
 * the result, and here everything is sorted.
 *
 * @param probabilities The output of softmax.
 * @param coin          A number in [0, 1).
 * @return The chosen token (not its position after sorting).
 */
std::size_t Sampler::sample_top_p(std::span<const float> probabilities, float coin) {
    // clear() keeps the capacity reserved by the constructor: no allocation here.
    candidates_.clear();
    for (std::size_t token = 0; token < probabilities.size(); ++token) {
        candidates_.push_back(Candidate{token, probabilities[token]});
    }
    std::sort(candidates_.begin(), candidates_.end(),
              [](const Candidate& left, const Candidate& right) {
                  return left.probability > right.probability;
              });

    // The nucleus ends at the first candidate whose cumulative sum passes top_p. If rounding
    // keeps the sum from ever passing it, the nucleus is every candidate.
    float nucleus_sum = 0.0f;
    std::size_t last_index = candidates_.size() - 1;
    for (std::size_t index = 0; index < candidates_.size(); ++index) {
        nucleus_sum += candidates_[index].probability;
        if (nucleus_sum > top_p_) {
            last_index = index;
            break;
        }
    }

    // The coin is in [0, 1) and the nucleus sums to nucleus_sum: scale one to the other.
    const float scaled_coin = coin * nucleus_sum;
    float running_sum = 0.0f;
    for (std::size_t index = 0; index <= last_index; ++index) {
        running_sum += candidates_[index].probability;
        if (scaled_coin < running_sum) {
            return candidates_[index].token;
        }
    }
    return candidates_[last_index].token; // rounding left the scaled coin above every sum
}

/**
 * @brief xorshift64*, as run.c's random_u32: three shift-XORs, then the upper half of a product.
 *
 * Unsigned overflow wraps around, which is defined, so the multiplication is the 64-bit
 * product modulo 2^64 on every platform.
 *
 * @return 32 random bits.
 */
std::uint32_t Sampler::next_random_u32() {
    random_state_ ^= random_state_ >> 12;
    random_state_ ^= random_state_ << 25;
    random_state_ ^= random_state_ >> 27;
    return static_cast<std::uint32_t>((random_state_ * 0x2545F4914F6CDD1D) >> 32);
}

/**
 * @brief A coin in [0, 1), as run.c's random_f32.
 *
 * Keeps the top 24 bits, the precision of a float, and divides by 2^24 = 16777216: the largest
 * value is (2^24 - 1) / 2^24, just below 1.
 *
 * @return The next coin.
 */
float Sampler::next_random_float() {
    const std::uint32_t top_24_bits = next_random_u32() >> 8;
    return static_cast<float>(top_24_bits) / 16777216.0f;
}

} // namespace tiny_llm
