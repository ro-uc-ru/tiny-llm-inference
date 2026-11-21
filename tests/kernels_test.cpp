#include <tiny_llm/kernels.hpp>
#include <tiny_llm/tensor.hpp>

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <vector>

namespace {

/**
 * Tolerance for float comparisons. The expected values are rounded to 6 decimals in Python,
 * so the tolerance must be somewhat larger than that rounding.
 */
constexpr float tolerance = 1e-5F;

/** @brief Adds element by element and leaves the addend untouched. */
TEST(Kernels, AddInPlace) {
    std::vector<float> accumulator{1.0F, 2.0F};
    const std::vector<float> addend{10.0F, 20.0F};

    tiny_llm::add_in_place(accumulator, addend);

    EXPECT_NEAR(accumulator[0], 11.0F, tolerance);
    EXPECT_NEAR(accumulator[1], 22.0F, tolerance);
    EXPECT_NEAR(addend[0], 10.0F, tolerance);
    EXPECT_NEAR(addend[1], 20.0F, tolerance);
}

/** @brief Spans of different sizes are a bug in the caller: the check aborts. */
TEST(KernelsDeathTest, AddInPlaceRejectsDifferentSizes) {
    std::vector<float> accumulator{1.0F, 2.0F};
    const std::vector<float> addend{10.0F};

    EXPECT_DEATH(tiny_llm::add_in_place(accumulator, addend),
                 "accumulator has 2 values, addend has 1");
}

/**
 * @brief silu(1) * 2 and silu(-1) * 2, with silu(value) = value / (1 + e^-value).
 *
 * silu is internal to kernels.cpp, so this test is also its test: a positive and a negative
 * value.
 */
TEST(Kernels, Swiglu) {
    std::vector<float> hidden{1.0F, -1.0F};
    const std::vector<float> gate{2.0F, 2.0F};

    tiny_llm::swiglu(hidden, gate);

    EXPECT_NEAR(hidden[0], 1.462117F, tolerance);
    EXPECT_NEAR(hidden[1], -0.537883F, tolerance);
}

/**
 * @brief A very negative value gives 0, not nan.
 *
 * e^100 overflows to infinity in float, and -100 / infinity is -0: the overflow is harmless
 * here, unlike in a softmax without the maximum subtracted.
 */
TEST(Kernels, SwigluOfVeryNegativeValueIsZero) {
    std::vector<float> hidden{-100.0F};
    const std::vector<float> gate{2.0F};

    tiny_llm::swiglu(hidden, gate);

    EXPECT_NEAR(hidden[0], 0.0F, tolerance);
}

/** @brief Spans of different sizes are a bug in the caller: the check aborts. */
TEST(KernelsDeathTest, SwigluRejectsDifferentSizes) {
    std::vector<float> hidden{1.0F, -1.0F};
    const std::vector<float> gate{2.0F};

    EXPECT_DEATH(tiny_llm::swiglu(hidden, gate), "hidden has 2 values, gate has 1");
}

/**
 * @brief mean(1, 4, 9, 16) = 7.5, sqrt(7.5 + 1e-5) = 2.738615, and each value is divided by it.
 *
 * The weights are all 1, so the result is the normalized input alone. A version that summed
 * the weights instead of the squares of the input would return the input unchanged.
 */
TEST(Kernels, Rmsnorm) {
    const std::vector<float> input{1.0F, 2.0F, 3.0F, 4.0F};
    const std::vector<float> weight{1.0F, 1.0F, 1.0F, 1.0F};
    std::vector<float> output(4);

    tiny_llm::rmsnorm(output, input, weight);

    EXPECT_NEAR(output[0], 0.365148F, tolerance);
    EXPECT_NEAR(output[1], 0.730296F, tolerance);
    EXPECT_NEAR(output[2], 1.095444F, tolerance);
    EXPECT_NEAR(output[3], 1.460593F, tolerance);
}

/** @brief Each output value is multiplied by its own weight. */
TEST(Kernels, RmsnormScalesByWeight) {
    const std::vector<float> input{1.0F, 2.0F, 3.0F, 4.0F};
    const std::vector<float> weight{2.0F, 1.0F, 0.0F, -1.0F};
    std::vector<float> output(4);

    tiny_llm::rmsnorm(output, input, weight);

    EXPECT_NEAR(output[0], 0.730296F, tolerance);
    EXPECT_NEAR(output[1], 0.730296F, tolerance);
    EXPECT_NEAR(output[2], 0.0F, tolerance);
    EXPECT_NEAR(output[3], -1.460593F, tolerance);
}

/**
 * @brief Output and input may be the same buffer, as kernels.hpp promises: the mean is
 * computed before any value is overwritten.
 */
TEST(Kernels, RmsnormInPlace) {
    std::vector<float> values{1.0F, 2.0F, 3.0F, 4.0F};
    const std::vector<float> weight{1.0F, 1.0F, 1.0F, 1.0F};

    tiny_llm::rmsnorm(values, values, weight);

    EXPECT_NEAR(values[0], 0.365148F, tolerance);
    EXPECT_NEAR(values[3], 1.460593F, tolerance);
}

/** @brief An empty input would divide by zero in the mean: the check aborts. */
TEST(KernelsDeathTest, RmsnormRejectsEmptyInput) {
    const std::vector<float> input;
    const std::vector<float> weight;
    std::vector<float> output;

    EXPECT_DEATH(tiny_llm::rmsnorm(output, input, weight), "input is empty");
}

/** @brief A weight of the wrong size is a bug in the caller: the check aborts. */
TEST(KernelsDeathTest, RmsnormRejectsDifferentSizes) {
    const std::vector<float> input{1.0F, 2.0F, 3.0F, 4.0F};
    const std::vector<float> weight{1.0F, 1.0F};
    std::vector<float> output(4);

    EXPECT_DEATH(tiny_llm::rmsnorm(output, input, weight),
                 "output has 4 values, input has 4, weight has 2");
}

/**
 * The 3 x 2 matrix of the guide, row after row:
 *
 *     row 0: 1 2
 *     row 1: 3 4
 *     row 2: 5 6
 */
constexpr std::array<float, 6> matmul_weight_values{1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F};

/** @brief (1·1 + 2·1, 3 + 4, 5 + 6) = {3, 7, 11}: one dot product per row. */
TEST(Kernels, Matmul) {
    const tiny_llm::Tensor weights{matmul_weight_values, {3, 2}};
    const std::vector<float> input{1.0F, 1.0F};
    std::vector<float> output(3);

    tiny_llm::matmul(output, weights, input);

    EXPECT_NEAR(output[0], 3.0F, tolerance);
    EXPECT_NEAR(output[1], 7.0F, tolerance);
    EXPECT_NEAR(output[2], 11.0F, tolerance);
}

/**
 * @brief An input with two different values: (2 - 2, 6 - 4, 10 - 6) = {0, 2, 4}.
 *
 * With {1, 1} every column weighs the same, so mixing up which input value goes with which
 * column would go unnoticed. Here it would not.
 */
TEST(Kernels, MatmulPairsEachColumnWithItsInput) {
    const tiny_llm::Tensor weights{matmul_weight_values, {3, 2}};
    const std::vector<float> input{2.0F, -1.0F};
    std::vector<float> output(3);

    tiny_llm::matmul(output, weights, input);

    EXPECT_NEAR(output[0], 0.0F, tolerance);
    EXPECT_NEAR(output[1], 2.0F, tolerance);
    EXPECT_NEAR(output[2], 4.0F, tolerance);
}

/** @brief An output that is not one value per row is a bug in the caller: the check aborts. */
TEST(KernelsDeathTest, MatmulRejectsWrongOutputSize) {
    const tiny_llm::Tensor weights{matmul_weight_values, {3, 2}};
    const std::vector<float> input{1.0F, 1.0F};
    std::vector<float> output(2);

    EXPECT_DEATH(tiny_llm::matmul(output, weights, input),
                 "output has 2 values, weights has 3 rows");
}

/** @brief An input that is not one value per column is a bug in the caller: the check aborts. */
TEST(KernelsDeathTest, MatmulRejectsWrongInputSize) {
    const tiny_llm::Tensor weights{matmul_weight_values, {3, 2}};
    const std::vector<float> input{1.0F, 1.0F, 1.0F};
    std::vector<float> output(3);

    EXPECT_DEATH(tiny_llm::matmul(output, weights, input),
                 "input has 3 values, weights has 2 columns");
}

/** @brief Maximum 3; e^-2, e^-1, e^0 over their sum 1.503215. */
TEST(Kernels, Softmax) {
    std::vector<float> values{1.0F, 2.0F, 3.0F};

    tiny_llm::softmax(values);

    EXPECT_NEAR(values[0], 0.090031F, tolerance);
    EXPECT_NEAR(values[1], 0.244728F, tolerance);
    EXPECT_NEAR(values[2], 0.665241F, tolerance);
}

/**
 * @brief Large scores do not overflow, thanks to subtracting the maximum.
 *
 * Without it, e^1000 is infinity in float and the result is `nan nan`. With it, the
 * exponentials are e^-1 and e^0, the same as for {0, 1}.
 */
TEST(Kernels, SoftmaxOfLargeValuesDoesNotOverflow) {
    std::vector<float> values{1000.0F, 1001.0F};

    tiny_llm::softmax(values);

    EXPECT_NEAR(values[0], 0.268941F, tolerance);
    EXPECT_NEAR(values[1], 0.731059F, tolerance);
}

/** @brief An empty span has no maximum: the check aborts. */
TEST(KernelsDeathTest, SoftmaxRejectsEmptyValues) {
    std::vector<float> values;

    EXPECT_DEATH(tiny_llm::softmax(values), "values is empty");
}

/**
 * @brief Position 1, head size 4: pair 0 turns 1 radian, pair 1 turns 0.01 radians.
 *
 * (1, 0) turned by an angle is (cos, sin) of that angle.
 */
TEST(Kernels, RotateRope) {
    std::vector<float> vector{1.0F, 0.0F, 1.0F, 0.0F};

    tiny_llm::rotate_rope(vector, 1, 4);

    EXPECT_NEAR(vector[0], 0.540302F, tolerance);
    EXPECT_NEAR(vector[1], 0.841471F, tolerance);
    EXPECT_NEAR(vector[2], 0.999950F, tolerance);
    EXPECT_NEAR(vector[3], 0.010000F, tolerance);
}

/**
 * @brief (0, 1) turned by an angle is (-sin, cos).
 *
 * This catches the sign of the sine and a pair written before both values were read:
 * overwriting the even value first would compute the odd one from the rotated value.
 */
TEST(Kernels, RotateRopeTurnsTheOddValueToo) {
    std::vector<float> vector{0.0F, 1.0F, 0.0F, 1.0F};

    tiny_llm::rotate_rope(vector, 1, 4);

    EXPECT_NEAR(vector[0], -0.841471F, tolerance);
    EXPECT_NEAR(vector[1], 0.540302F, tolerance);
    EXPECT_NEAR(vector[2], -0.010000F, tolerance);
    EXPECT_NEAR(vector[3], 0.999950F, tolerance);
}

/**
 * @brief Two heads of 4: the second head turns exactly like the first.
 *
 * The angle depends on the pair inside its head. A version that used the place in the whole
 * vector would turn pair (4, 5) by theta_2 instead of theta_0.
 */
TEST(Kernels, RotateRopeTurnsEveryHeadTheSame) {
    std::vector<float> vector{1.0F, 0.0F, 1.0F, 0.0F, 1.0F, 0.0F, 1.0F, 0.0F};

    tiny_llm::rotate_rope(vector, 1, 4);

    EXPECT_NEAR(vector[4], 0.540302F, tolerance);
    EXPECT_NEAR(vector[5], 0.841471F, tolerance);
    EXPECT_NEAR(vector[6], 0.999950F, tolerance);
    EXPECT_NEAR(vector[7], 0.010000F, tolerance);
}

/** @brief At position 0 every angle is 0, and nothing moves. */
TEST(Kernels, RotateRopeAtPositionZeroChangesNothing) {
    std::vector<float> vector{1.0F, 2.0F, 3.0F, 4.0F};

    tiny_llm::rotate_rope(vector, 0, 4);

    EXPECT_NEAR(vector[0], 1.0F, tolerance);
    EXPECT_NEAR(vector[1], 2.0F, tolerance);
    EXPECT_NEAR(vector[2], 3.0F, tolerance);
    EXPECT_NEAR(vector[3], 4.0F, tolerance);
}

/**
 * @brief A rotation keeps the length: (3, 4) has length 5 before and after.
 *
 * Head size 2 has a single pair with theta 1, so at position 5 it turns 5 radians, to
 * (4.686684, -1.742124).
 */
TEST(Kernels, RotateRopeKeepsTheLengthOfEachPair) {
    std::vector<float> vector{3.0F, 4.0F};

    tiny_llm::rotate_rope(vector, 5, 2);

    EXPECT_NEAR(vector[0], 4.686684F, tolerance);
    EXPECT_NEAR(vector[1], -1.742124F, tolerance);
    EXPECT_NEAR(std::hypot(vector[0], vector[1]), 5.0F, tolerance);
}

/** @brief An odd head size cannot be cut into pairs: the check aborts. */
TEST(KernelsDeathTest, RotateRopeRejectsOddHeadSize) {
    std::vector<float> vector{1.0F, 0.0F, 1.0F};

    EXPECT_DEATH(tiny_llm::rotate_rope(vector, 1, 3), "head size must be even and positive, got 3");
}

/** @brief A vector that is not a whole number of heads is a bug in the caller: the check aborts. */
TEST(KernelsDeathTest, RotateRopeRejectsPartialHead) {
    std::vector<float> vector{1.0F, 0.0F, 1.0F, 0.0F, 1.0F, 0.0F};

    EXPECT_DEATH(tiny_llm::rotate_rope(vector, 1, 4),
                 "vector has 6 values, not a whole number of heads of 4");
}

} // namespace

/**
 * @brief The toy of step 4: keys (1, 0) and (0, 1), values (1, 2) and (3, 4), query (1, 0).
 *
 * Scores (1/√2, 0) = (0.707107, 0); softmax (0.669761, 0.330239);
 * output 0.669761 · (1, 2) + 0.330239 · (3, 4) = (1.660478, 2.660478).
 */
TEST(Attend, ToyExample) {
    const std::vector<float> query{1.0F, 0.0F};
    const std::vector<float> keys{1.0F, 0.0F, 0.0F, 1.0F};
    const std::vector<float> values{1.0F, 2.0F, 3.0F, 4.0F};
    std::vector<float> scores(2);
    std::vector<float> output(2);

    tiny_llm::attend(output, query, keys, values, scores, 2, 2);

    EXPECT_NEAR(scores[0], 0.669761F, tolerance);
    EXPECT_NEAR(scores[1], 0.330239F, tolerance);
    EXPECT_NEAR(output[0], 1.660478F, tolerance);
    EXPECT_NEAR(output[1], 2.660478F, tolerance);
}

/**
 * @brief The same toy with a second KV head in between (kv_stride 4), filled with 9s.
 *
 * Layout of the cache: position 0 = {head 0: 1 0, head 1: 9 9}, position 1 = {0 1, 9 9}.
 * If the stride were ignored, the 9s would enter the scores and the output would change.
 */
TEST(Attend, SkipsTheOtherKvHeads) {
    const std::vector<float> query{1.0F, 0.0F};
    const std::vector<float> keys{1.0F, 0.0F, 9.0F, 9.0F, 0.0F, 1.0F, 9.0F, 9.0F};
    const std::vector<float> values{1.0F, 2.0F, 9.0F, 9.0F, 3.0F, 4.0F, 9.0F, 9.0F};
    std::vector<float> scores(2);
    std::vector<float> output(2);

    tiny_llm::attend(output, query, keys, values, scores, 2, 4);

    EXPECT_NEAR(output[0], 1.660478F, tolerance);
    EXPECT_NEAR(output[1], 2.660478F, tolerance);
}

/** @brief At position 0 there is one key: its weight is 1 and the output is its value. */
TEST(Attend, OnePositionCopiesItsValue) {
    const std::vector<float> query{1.0F, 0.0F};
    const std::vector<float> keys{0.5F, -3.0F};
    const std::vector<float> values{7.0F, 8.0F};
    std::vector<float> scores{-1.0F, -1.0F};
    std::vector<float> output{100.0F, 100.0F};

    tiny_llm::attend(output, query, keys, values, scores, 1, 2);

    EXPECT_NEAR(output[0], 7.0F, tolerance);
    EXPECT_NEAR(output[1], 8.0F, tolerance);
    EXPECT_NEAR(scores[1], -1.0F, tolerance); // beyond position_count: untouched
}

/** @brief Two positions need (2 − 1) · 2 + 2 = 4 floats of keys; 3 is a bug in the caller. */
TEST(AttendDeathTest, RejectsKeysThatAreTooShort) {
    const std::vector<float> query{1.0F, 0.0F};
    const std::vector<float> keys{1.0F, 0.0F, 0.0F};
    const std::vector<float> values{1.0F, 2.0F, 3.0F, 4.0F};
    std::vector<float> scores(2);
    std::vector<float> output(2);

    EXPECT_DEATH(tiny_llm::attend(output, query, keys, values, scores, 2, 2),
                 "keys has 3 and values 4 floats, 2 positions need 4");
}
