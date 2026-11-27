#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <string_view>

namespace tiny_llm {

/**
 * @brief Command-line options, with the flags and defaults of llama2.c's run.c.
 *
 * Only generation is supported: run.c's -m (mode) and -y (system prompt) are unknown flags.
 */
struct Options {
    std::filesystem::path checkpoint_path;                  /**< The first argument. */
    std::filesystem::path tokenizer_path = "tokenizer.bin"; /**< -z */
    float temperature = 1.0f; /**< -t; negative becomes 0, as in run.c. */
    float top_p = 0.9f;       /**< -p; outside [0, 1] becomes 0.9, as in run.c. */
    std::size_t steps = 256;  /**< -n; 0 or above the context means the whole context. */
    std::uint64_t seed = 0;   /**< -s; 0 means take it from the clock, as run.c does. */
    std::string prompt;       /**< -i; empty by default. */
};

/** @brief The usage line, printed after every usage error. */
inline constexpr std::string_view usage =
    "usage: tiny_llm <checkpoint.bin> [-t temperature] [-p top_p] [-n steps] [-s seed] "
    "[-i prompt] [-z tokenizer.bin]";

/**
 * @brief Reads a whole float, rejecting trailing characters ("0.9x").
 *
 * std::stof alone reads "0.9x" as 0.9 and says nothing; the count of parsed characters
 * catches what is left over.
 *
 * @param flag The flag the value belongs to, for the error message.
 * @param text The value as written on the command line.
 * @return The float `text` spells.
 * @throws std::invalid_argument If `text` is not a number or has characters left over.
 * @throws std::out_of_range     If it does not fit in a float.
 */
[[nodiscard]] float parse_float(std::string_view flag, const std::string& text);

/**
 * @brief Reads a whole non-negative integer: it must start with a digit (no "-5", " 5" or
 * "+5") and have nothing left over (no "1e9").
 *
 * std::stoul alone wraps "-5" around to 18446744073709551611 and reads "1e9" as 1.
 *
 * @param flag The flag the value belongs to, for the error message.
 * @param text The value as written on the command line.
 * @return The integer `text` spells.
 * @throws std::invalid_argument If `text` is not a non-negative integer.
 * @throws std::out_of_range     If it does not fit in 64 bits.
 */
[[nodiscard]] std::uint64_t parse_count(std::string_view flag, const std::string& text);

/**
 * @brief Parses `tiny_llm <checkpoint> [-t temperature] [-p top_p] [-n steps] [-s seed]
 * [-i prompt] [-z tokenizer]`.
 *
 * After the checkpoint, arguments come in pairs: a flag and its value. A flag given twice
 * keeps the last value, as in run.c. Applies run.c's two corrections at the end: a negative
 * temperature becomes 0 and a top_p outside [0, 1] becomes 0.9.
 *
 * Example: {"tiny_llm", "model.bin", "-t", "0", "-n", "5"} gives checkpoint_path "model.bin",
 * temperature 0, steps 5 and the defaults for the rest.
 *
 * @param arguments The whole argv, program name included.
 * @return The options, with run.c's defaults for every flag not given.
 * @throws std::invalid_argument On a missing checkpoint, an unknown flag, a flag without a
 *                               value or a badly written number.
 * @throws std::out_of_range     On a number too large for its type.
 */
[[nodiscard]] Options parse_options(std::span<char*> arguments);

} // namespace tiny_llm
