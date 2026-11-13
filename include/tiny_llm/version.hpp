#pragma once

#include <string_view>

namespace tiny_llm {

// The library version, as set in CMakeLists.txt (project(... VERSION ...)).
[[nodiscard]] std::string_view version() noexcept;

} // namespace tiny_llm
