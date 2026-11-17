#pragma once

#include <string_view>

namespace tiny_llm {

/**
 * @brief The library version, as set in CMakeLists.txt (`project(... VERSION ...)`).
 * @return A view of a string literal, valid for the whole program.
 */
[[nodiscard]] std::string_view version() noexcept;

} // namespace tiny_llm
