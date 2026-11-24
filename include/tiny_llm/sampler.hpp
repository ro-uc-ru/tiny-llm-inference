#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace tiny_llm {

/**
 * @brief Picks the next token from the logits: argmax, multinomial or top-p.
 *
 * | temperature | top_p           | What sample() does                                          |
 * | ----------- | --------------- | ----------------------------------------------------------- |
 * | 0           | ignored         | The token with the highest logit; draws no random number.   |
 * | > 0         | 0 or >= 1       | logits / temperature, softmax, multinomial with one coin.   |
 * | > 0         | between 0 and 1 | The same, but only among the most likely tokens summing top_p. |
 *
 * The coin comes from the xorshift generator of llama2.c's run.c, not from `<random>`: with the
 * same seed it gives the same numbers, so the generated text can be compared with run.c at a
 * temperature other than 0.
 *
 * Only sample() is public: the generator and the two sampling methods are tested through it,
 * with a fixed seed (see tests/sampler_test.cpp).
 */
class Sampler {
  public:
    /**
     * @brief Stores the settings and reserves the top-p candidates once.
     *
     * Reserving here means sample() never allocates, not even on its first top-p call.
     * A zero seed is a bug in the caller and aborts with TINY_LLM_CHECK: xorshift of 0 is 0
     * forever, so every coin would be 0.
     *
     * @param vocab_size  The number of logits sample() will receive (Config::vocab_size).
     * @param temperature 0 for argmax; larger values flatten the distribution.
     * @param top_p       The nucleus size in (0, 1); 0 or >= 1 turns top-p off.
     * @param seed        The initial xorshift state; must not be 0.
     */
    Sampler(std::size_t vocab_size, float temperature, float top_p, std::uint64_t seed);

    /**
     * @brief Chooses the next token.
     *
     * With temperature 0, returns the argmax and leaves `logits` untouched. Otherwise divides
     * the logits by the temperature, turns them into probabilities with softmax (in place),
     * draws one coin and samples with multinomial or top-p.
     *
     * An empty span is a bug in the caller and aborts with TINY_LLM_CHECK.
     *
     * @param logits The logits of the forward pass; overwritten with probabilities when
     *               temperature > 0.
     * @return The chosen token, an index into `logits`.
     */
    [[nodiscard]] std::size_t sample(std::span<float> logits);

  private:
    /** @brief A token and its probability, for sorting in top-p (run.c's ProbIndex). */
    struct Candidate {
        std::size_t token;   /**< The index into the probabilities. */
        float probability;   /**< Its probability after softmax. */
    };

    /**
     * @brief Top-p (nucleus) sampling.
     *
     * Sorts the tokens from most to least likely, keeps the smallest prefix whose sum exceeds
     * top_p (the nucleus), scales the coin by that sum and returns the first token of the
     * nucleus whose cumulative probability exceeds the scaled coin. If rounding leaves the coin
     * above every cumulative value, returns the last token of the nucleus.
     *
     * @param probabilities The output of softmax; sums to 1.
     * @param coin          A number in [0, 1).
     * @return The chosen token, an index into `probabilities`.
     */
    [[nodiscard]] std::size_t sample_top_p(std::span<const float> probabilities, float coin);

    /**
     * @brief Advances the xorshift state and returns 32 random bits.
     *
     * `state ^= state >> 12; state ^= state << 25; state ^= state >> 27`, then the upper 32 bits
     * of `state * 0x2545F4914F6CDD1D`. Unsigned overflow wraps around, which is defined.
     * With seed 42 the first three values are 1456360119, 3359763283 and 3393612768.
     *
     * @return The next 32-bit random number.
     */
    [[nodiscard]] std::uint32_t next_random_u32();

    /**
     * @brief A random float in [0, 1): the top 24 bits of next_random_u32() divided by 2^24.
     *
     * 24 bits is the precision of a float. With seed 42 the first three values are 0.339085,
     * 0.782256 and 0.790137.
     *
     * @return The next coin.
     */
    [[nodiscard]] float next_random_float();

    float temperature_;                 /**< 0 means argmax. */
    float top_p_;                       /**< Top-p is on only for values in (0, 1). */
    std::uint64_t random_state_;        /**< The xorshift state; never 0. */
    std::vector<Candidate> candidates_; /**< Reused by top-p, reserved to vocab_size. */
};

} // namespace tiny_llm
