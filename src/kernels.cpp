#include <tiny_llm/check.hpp>
#include <tiny_llm/kernels.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <format>
#include <span>

namespace tiny_llm {
namespace {

/**
 * @brief The SiLU activation (also called swish): `value * sigmoid(value)`.
 *
 * Internal to this file: the unnamed namespace gives it internal linkage, like `static` in C,
 * so it cannot clash with another `silu` elsewhere in the program.
 *
 * @param value Any finite float.
 * @return `value / (1 + e^-value)`.
 */
float silu(float value) {
    return value / (1.0f + std::exp(-value));
}

} // namespace

void add_in_place(std::span<float> accumulator, std::span<const float> addend) {
    TINY_LLM_CHECK(
        accumulator.size() == addend.size(),
        std::format("accumulator has {} values, addend has {}", accumulator.size(), addend.size()));
    for (std::size_t index = 0; index < accumulator.size(); index++) {
        accumulator[index] += addend[index];
    }
}

void swiglu(std::span<float> hidden, std::span<const float> gate) {
    TINY_LLM_CHECK(hidden.size() == gate.size(),
                   std::format("hidden has {} values, gate has {}", hidden.size(), gate.size()));
    for (std::size_t index = 0; index < hidden.size(); index++) {
        hidden[index] = silu(hidden[index]) * gate[index];
    }
}

void rmsnorm(std::span<float> output, std::span<const float> input, std::span<const float> weight) {
    TINY_LLM_CHECK(!input.empty(), "input is empty: the mean would divide by zero");
    TINY_LLM_CHECK(output.size() == input.size() && weight.size() == input.size(),
                   std::format("output has {} values, input has {}, weight has {}", output.size(),
                               input.size(), weight.size()));

    float sum_of_squares = 0.0f;
    for (auto value : input) {
        sum_of_squares += value * value;
    }
    // Epsilon goes inside the root, after the mean, so the root is never zero.
    const float root_mean_square =
        std::sqrt(sum_of_squares / static_cast<float>(input.size()) + rmsnorm_epsilon);
    // One division here, then a multiplication per element instead of a division.
    const float inverse_root_mean_square = 1.0f / root_mean_square;

    for (std::size_t index = 0; index < output.size(); index++) {
        output[index] = weight[index] * input[index] * inverse_root_mean_square;
    }
}

void matmul(std::span<float> output, const Tensor& weights, std::span<const float> input) {
    TINY_LLM_CHECK(
        output.size() == weights.rows(),
        std::format("output has {} values, weights has {} rows", output.size(), weights.rows()));
    TINY_LLM_CHECK(input.size() == weights.columns(),
                   std::format("input has {} values, weights has {} columns", input.size(),
                               weights.columns()));

    // One dot product per row: the row of weights against the whole input.
    for (std::size_t row_index = 0; row_index < weights.rows(); row_index++) {
        const std::span<const float> row = weights.row(row_index);
        float sum = 0.0f;
        for (std::size_t column_index = 0; column_index < input.size(); column_index++) {
            sum += row[column_index] * input[column_index];
        }
        output[row_index] = sum;
    }
}

void softmax(std::span<float> values) {
    TINY_LLM_CHECK(!values.empty(),
                   "values is empty, softmax makes no sense when there is no value");
    const float max_value = *std::max_element(values.begin(), values.end());

    // Each exponential is computed once and stored in place; the second loop only divides.
    float exponential_sum = 0.0f;
    for (auto& value : values) {
        value = std::exp(value - max_value);
        exponential_sum += value;
    }
    for (auto& value : values) {
        value /= exponential_sum;
    }
}

void rotate_rope(std::span<float> vector, std::size_t position, std::size_t head_size) {
    TINY_LLM_CHECK(head_size > 0 && head_size % 2 == 0,
                   std::format("head size must be even and positive, got {}", head_size));
    TINY_LLM_CHECK(vector.size() % head_size == 0,
                   std::format("vector has {} values, not a whole number of heads of {}",
                               vector.size(), head_size));

    const std::size_t head_count = vector.size() / head_size;
    const float position_as_float = static_cast<float>(position);
    const float head_size_as_float = static_cast<float>(head_size);

    for (std::size_t head_index = 0; head_index < head_count; head_index++) {
        const std::size_t head_start = head_index * head_size;
        for (std::size_t pair_index = 0; pair_index < head_size / 2; pair_index++) {
            // theta depends on the pair inside the head, not on its place in the vector.
            const float exponent = -2.0f * static_cast<float>(pair_index) / head_size_as_float;
            const float theta = std::pow(10000.0f, exponent);
            const float angle = position_as_float * theta;
            const float cosine = std::cos(angle);
            const float sine = std::sin(angle);

            // Read both values before writing: the second one needs the first unrotated.
            const std::size_t even_index = head_start + 2 * pair_index;
            const std::size_t odd_index = even_index + 1;
            const float even_value = vector[even_index];
            const float odd_value = vector[odd_index];
            vector[even_index] = even_value * cosine - odd_value * sine;
            vector[odd_index] = even_value * sine + odd_value * cosine;
        }
    }
}

} // namespace tiny_llm
