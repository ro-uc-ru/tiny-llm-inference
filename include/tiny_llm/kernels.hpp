#pragma once

#include <tiny_llm/tensor.hpp>

#include <cstddef>
#include <span>

namespace tiny_llm {
/**
 * @brief RMSNorm epsilon, as in llama2.c.
 *
 * Added to the mean of the squares before the square root, so the divisor is never zero,
 * not even for an all-zero input.
 */
inline constexpr float rmsnorm_epsilon = 1e-5f;

/**
 * @brief Adds `addend` to `accumulator`, element by element: the residual connection.
 *
 * `accumulator[index] += addend[index]` for every index. The forward pass uses it as
 * `hidden += projection`, after attention and after the feed-forward network.
 *
 * Both spans must have the same size; a mismatch is a bug in the caller and aborts with
 * TINY_LLM_CHECK.
 *
 * @param accumulator Read and overwritten with the sum.
 * @param addend The values added; not modified.
 */
void add_in_place(std::span<float> accumulator, std::span<const float> addend);

/**
 * @brief The SwiGLU gate of the feed-forward network, in place.
 *
 * `hidden[index] = silu(hidden[index]) * gate[index]`, with `silu(value) = value / (1 + e^-value)`.
 * `hidden` comes from the w1 projection and `gate` from w3; the result goes into w2.
 *
 * Both spans must have the same size; a mismatch is a bug in the caller and aborts with
 * TINY_LLM_CHECK.
 *
 * @param hidden Read and overwritten with the gated values.
 * @param gate The gate values; not modified.
 */
void swiglu(std::span<float> hidden, std::span<const float> gate);

/**
 * @brief Root-mean-square normalization, scaled by a learned weight per element.
 *
 * `output[index] = weight[index] * input[index] / sqrt(mean(input²) + rmsnorm_epsilon)`.
 * The forward pass applies it before attention, before the feed-forward network and once
 * more before the classifier.
 *
 * The three spans must have the same size and `input` must not be empty (the mean would
 * divide by zero); a violation is a bug in the caller and aborts with TINY_LLM_CHECK.
 * `output` and `input` may be the same buffer: every input value is read for the mean
 * before any output value is written, and each output value then depends only on the
 * input value at its own index.
 *
 * @param output Overwritten with the normalized values.
 * @param input The values to normalize; not modified.
 * @param weight The learned scale, one per element (`rms_att_weight`, `rms_ffn_weight` or
 *        `rms_final_weight` in llama2.c); not modified.
 */
void rmsnorm(std::span<float> output, std::span<const float> input, std::span<const float> weight);

/**
 * @brief Matrix-vector product: `output = weights · input`.
 *
 * `output[row] = sum over columns of weights(row, column) * input[column]`. `weights` has
 * one row per output value and one column per input value, so `rows()` is the output size
 * and `columns()` the input size. The forward pass calls it seven times per layer (the
 * query, key, value and output projections, and w1, w2 and w3) and once more for the
 * classifier. It is where almost all the time goes, and block 3 optimizes it.
 *
 * Sizes that do not match the shape of `weights` are a bug in the caller and abort with
 * TINY_LLM_CHECK. `output` must not overlap `input`: every output value reads the whole input.
 *
 * @param output Overwritten with the product; `weights.rows()` values.
 * @param weights The matrix, row-major; a view into the checkpoint, not modified.
 * @param input The vector; `weights.columns()` values, not modified.
 */
void matmul(std::span<float> output, const Tensor& weights, std::span<const float> input);

/**
 * @brief Softmax in place: turns scores into probabilities that add up to 1.
 *
 * `values[index] = e^(values[index] - maximum) / sum over all of e^(value - maximum)`.
 * Subtracting the maximum first does not change the result and keeps `e^` from overflowing
 * to infinity: without it, `{1000, 1001}` gives `nan nan`. Used on the attention scores of
 * each head and, by the sampler, on the logits.
 *
 * An empty span is a bug in the caller and aborts with TINY_LLM_CHECK: it has no maximum.
 *
 * @param values Read and overwritten with the probabilities.
 */
void softmax(std::span<float> values);

/**
 * @brief Rotary position embedding (RoPE), in place, for every head of `vector`.
 *
 * `vector` is cut into heads of `head_size` values, and each head into pairs
 * `(2 * pair, 2 * pair + 1)`. Each pair turns by the angle `position * theta(pair)`, with
 * `theta(pair) = 10000^(-2 * pair / head_size)`; the angle depends on the pair inside its
 * head, not on its place in `vector`. The forward pass calls it once for the query (all the
 * heads) and once for the key (only the key-value heads).
 *
 * A `head_size` that is zero or odd, or a `vector` that is not a whole number of heads, is a
 * bug in the caller and aborts with TINY_LLM_CHECK.
 *
 * @param vector Read and overwritten with the rotated values.
 * @param position Position of the token in the sequence, from 0.
 * @param head_size Values per head; even and greater than zero.
 */
void rotate_rope(std::span<float> vector, std::size_t position, std::size_t head_size);

/**
 * @brief Attention of one query head over positions 0 … position_count − 1 of one KV head.
 *
 * `score[t] = query · key[t] / sqrt(head_size)`, then softmax over the scores, then
 * `output = sum over t of score[t] * value[t]`. The head size is `query.size()`.
 *
 * `keys` and `values` start at the chosen KV head of position 0 in one layer's cache. The
 * key of position t starts `t * kv_stride` floats further on: in the cache, the other KV
 * heads of position 0 sit between position 0 and position 1 of this head. The forward pass
 * calls it once per query head, so it is tested on its own, like the other kernels.
 *
 * Sizes that do not fit are a bug in the caller and abort with TINY_LLM_CHECK.
 *
 * @param output Overwritten with the mixed values; `query.size()` floats.
 * @param query The query of this head, already rotated by RoPE; not modified.
 * @param keys The cache from this KV head of position 0 on; not modified.
 * @param values Same layout as `keys`; not modified.
 * @param scores Work buffer; at least `position_count` floats, the first ones overwritten.
 * @param position_count How many positions to attend to: the current position plus one.
 * @param kv_stride Floats from one position to the next in the cache: kv_head_count * head_size.
 */
void attend(std::span<float> output, std::span<const float> query, std::span<const float> keys,
            std::span<const float> values, std::span<float> scores, std::size_t position_count,
            std::size_t kv_stride);

} // namespace tiny_llm
