#include <tiny_llm/check.hpp>
#include <tiny_llm/load_error.hpp>
#include <tiny_llm/weights.hpp>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <format>
#include <span>
#include <utility>
#include <vector>

namespace tiny_llm {

/**
 * @brief How many floats follow the header: one term per section of the file, in file order.
 *
 * Every term is rows * columns of one matrix, times layer_count when there is one per layer.
 *
 * @param config A config already validated by parse_config.
 * @return The number of floats after the header.
 */
std::size_t weight_float_count(const Config& config) {
    const std::size_t head_size = config.embedding_size / config.head_count;
    const std::size_t kv_size =
        config.kv_head_count * head_size; // rows of K and V; < embedding_size with GQA
    const std::size_t layer_count = config.layer_count;

    const std::size_t token_embedding = config.vocab_size * config.embedding_size;
    const std::size_t attention_norm = layer_count * config.embedding_size;
    const std::size_t query_projection =
        layer_count * config.embedding_size * config.embedding_size;
    const std::size_t key_projection = layer_count * kv_size * config.embedding_size;
    const std::size_t value_projection = layer_count * kv_size * config.embedding_size;
    const std::size_t attention_output =
        layer_count * config.embedding_size * config.embedding_size;
    const std::size_t ffn_norm = layer_count * config.embedding_size;
    const std::size_t feed_forward_hidden =
        layer_count * config.feed_forward_size * config.embedding_size;
    const std::size_t feed_forward_output =
        layer_count * config.embedding_size * config.feed_forward_size;
    const std::size_t feed_forward_gate =
        layer_count * config.feed_forward_size * config.embedding_size;
    const std::size_t final_norm = config.embedding_size;
    // llama2.c still writes the RoPE tables (cos and sin, head_size / 2 each per position).
    const std::size_t rope_tables = config.max_sequence_length * head_size;
    const std::size_t classifier =
        config.shares_classifier_weights ? 0 : config.vocab_size * config.embedding_size;

    return token_embedding + attention_norm + query_projection + key_projection + value_projection +
           attention_output + ffn_norm + feed_forward_hidden + feed_forward_output +
           feed_forward_gate + final_norm + rope_tables + classifier;
}

/**
 * @brief Slices the weight section into named views, in file order.
 *
 * Four phases: check the size against the config (LoadError: the file comes from outside),
 * view the bytes as floats, walk them with a cursor, and check the cursor ended exactly at
 * the end (TINY_LLM_CHECK: after the size check, a mismatch is a bug in this walk).
 *
 * @param file_bytes The whole file, header included. Must outlive the result.
 * @param config     The header, already parsed by parse_config.
 * @return One Tensor per matrix; none of them owns its data.
 * @throws LoadError If the file size does not match the config.
 */
Weights map_weights(std::span<const std::byte> file_bytes, const Config& config) {
    // Phase 1: the file must be exactly as long as the config says. No subtraction, so
    // nothing can wrap around below zero.
    const std::size_t float_count = weight_float_count(config);
    const std::size_t expected_byte_count = header_size_in_bytes + sizeof(float) * float_count;
    if (file_bytes.size() != expected_byte_count) {
        throw LoadError(std::format("the config needs a file of {} bytes, got {}",
                                    expected_byte_count, file_bytes.size()));
    }

    // Phase 2: the same bytes, seen as floats. No copy.
    const std::byte* const first_float_byte = file_bytes.data() + header_size_in_bytes;
    TINY_LLM_CHECK(reinterpret_cast<std::uintptr_t>(first_float_byte) % alignof(float) == 0,
                   "the float data must be aligned to 4 bytes");
    const std::span<const float> floats{reinterpret_cast<const float*>(first_float_byte),
                                        float_count};

    // Phase 3: a cursor over the floats. Both lambdas capture it by reference, so every
    // call continues where the previous one stopped.
    std::size_t offset = 0;
    auto take = [&](std::size_t rows, std::size_t columns) -> Tensor {
        const Tensor tensor{floats.subspan(offset, rows * columns), {rows, columns}};
        offset += rows * columns;
        return tensor;
    };
    auto take_per_layer = [&](std::size_t rows, std::size_t columns) -> std::vector<Tensor> {
        std::vector<Tensor> tensors;
        tensors.reserve(config.layer_count);
        for (std::size_t layer = 0; layer < config.layer_count; ++layer) {
            tensors.push_back(take(rows, columns));
        }
        return tensors;
    };

    const std::size_t embedding_size = config.embedding_size;
    const std::size_t feed_forward_size = config.feed_forward_size;
    const std::size_t head_size = embedding_size / config.head_count;
    const std::size_t kv_size = config.kv_head_count * head_size;

    // In file order; see the table of the guide.
    Tensor token_embedding = take(config.vocab_size, embedding_size);
    std::vector<Tensor> attention_norm = take_per_layer(embedding_size, 1);
    std::vector<Tensor> query_projection = take_per_layer(embedding_size, embedding_size);
    std::vector<Tensor> key_projection = take_per_layer(kv_size, embedding_size);
    std::vector<Tensor> value_projection = take_per_layer(kv_size, embedding_size);
    std::vector<Tensor> attention_output = take_per_layer(embedding_size, embedding_size);
    std::vector<Tensor> ffn_norm = take_per_layer(embedding_size, 1);
    std::vector<Tensor> feed_forward_hidden = take_per_layer(feed_forward_size, embedding_size);
    std::vector<Tensor> feed_forward_output = take_per_layer(embedding_size, feed_forward_size);
    std::vector<Tensor> feed_forward_gate = take_per_layer(feed_forward_size, embedding_size);
    Tensor final_norm = take(embedding_size, 1);
    offset += config.max_sequence_length * head_size; // RoPE tables: in the file, never read
    Tensor classifier = config.shares_classifier_weights
                            ? token_embedding // a copy of the view, not of the data
                            : take(config.vocab_size, embedding_size);

    // Phase 4: the size was checked in phase 1, so ending anywhere else is a bug here.
    TINY_LLM_CHECK(offset == floats.size(),
                   std::format("the walk ended at float {} of {}", offset, floats.size()));

    // std::move hands over each vector's heap buffer instead of copying it.
    return Weights{token_embedding,
                   std::move(attention_norm),
                   std::move(query_projection),
                   std::move(key_projection),
                   std::move(value_projection),
                   std::move(attention_output),
                   std::move(ffn_norm),
                   std::move(feed_forward_hidden),
                   std::move(feed_forward_output),
                   std::move(feed_forward_gate),
                   final_norm,
                   classifier};
}

} // namespace tiny_llm
