#pragma once

#include <source_location>
#include <string_view>

/**
 * @brief The guard for invariants, that is, for bugs in this code.
 *
 * If `condition` is false, prints to stderr the condition as written in the source, the
 * message, and the file, line and function of the check, then calls `std::abort()`.
 * Destructors do not run.
 *
 * Always on, also in release builds (unlike `assert`, which disappears with `NDEBUG`):
 * checking a shape costs nothing next to a matmul. Environment errors (a missing or truncated
 * file) are not checks: they throw tiny_llm::LoadError. See docs/error-handling.md.
 *
 * Why a macro and not a function:
 * - `#condition` turns the condition into text ("head_count > 0"). A function would only
 *   receive its value, `false`.
 * - `message` is evaluated only when the check fails, because it sits behind the `if`.
 *   A message built with `std::format` costs nothing while the check passes.
 *
 * How the expansion is protected:
 * - `do { ... } while (0)` makes the macro a single statement that needs its `;`, so an
 *   `else` after it cannot attach to the macro's own `if` (the "dangling else").
 * - `!(condition)` keeps the parentheses: without them, `a || b` would expand to `!a || b`.
 * - `::tiny_llm::check_failed` is fully qualified, because the macro expands in the caller's
 *   code, which may be outside the namespace. Macros ignore namespaces, hence the prefix.
 * - `std::source_location::current()` is written inside the macro, so it records the line
 *   where the check is used, not a line in this header.
 *
 * @param condition An expression convertible to `bool`; evaluated exactly once.
 * @param message Anything convertible to `std::string_view`; evaluated only on failure.
 *
 * Example:
 * @code
 * TINY_LLM_CHECK(row_index < row_count, "row index out of range");
 * @endcode
 */
#define TINY_LLM_CHECK(condition, message)                                                         \
    do {                                                                                           \
        if (!(condition)) {                                                                        \
            ::tiny_llm::check_failed(#condition, (message), std::source_location::current());      \
        }                                                                                          \
    } while (0)

namespace tiny_llm {

/**
 * @brief Prints the failed check to stderr and calls `std::abort()`.
 *
 * Called only by TINY_LLM_CHECK. `[[noreturn]]`: the compiler knows that this call never
 * comes back, so a function that ends in a failed check needs no `return` after it, and no
 * code is generated after the call.
 *
 * `std::string_view` is a non-owning view of text (pointer and length), the `span` of strings:
 * it accepts both a string literal and a `std::string`, without copying either.
 *
 * @param condition The failed condition, as written in the source.
 * @param message Why the condition matters.
 * @param location Where the check is, filled in by the macro at the call site.
 */
[[noreturn]] void check_failed(std::string_view condition, std::string_view message,
                               std::source_location location);

} // namespace tiny_llm
