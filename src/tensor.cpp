#include <tiny_llm/check.hpp>
#include <tiny_llm/tensor.hpp>

#include <array>
#include <format>
#include <span>

namespace tiny_llm {

/**
 * @brief Wraps `data` as a `rows x columns` matrix, without copying.
 *
 * The members are built in the initializer list, in declaration order, and the check runs
 * on them afterwards: if it fails the program aborts, so it does not matter that they were
 * already stored. A shape that does not match the size is a bug in the code that sliced the
 * weights; a truncated file has been rejected before, with LoadError.
 *
 * @param data The values, row after row. Must outlive the Tensor.
 * @param shape `{rows, columns}`.
 */
Tensor::Tensor(std::span<const float> data, std::array<std::size_t, 2> shape)
    : data_(data), rows_(shape[0]), columns_(shape[1]) {
    TINY_LLM_CHECK(rows_ * columns_ == data_.size(),
                   std::format("shape {}x{} needs {} values, got {}", rows_, columns_,
                               rows_ * columns_, data_.size()));
}
/**
 * @brief One value, with both indices checked.
 *
 * An index out of range is a bug in the caller, not a problem with the file, so it aborts
 * with TINY_LLM_CHECK instead of throwing. Each index has its own check, so the report says
 * which one failed.
 *
 * @param row Row index, less than rows().
 * @param column Column index, less than columns().
 * @return The value at `data[row * columns() + column]`.
 */
float Tensor::at(std::size_t row, std::size_t column) const {
    TINY_LLM_CHECK(row < rows_, std::format("row {} out of range, rows: {}", row, rows_));
    TINY_LLM_CHECK(column < columns_,
                   std::format("column {} out of range, columns: {}", column, columns_));
    return data_[row * columns_ + column];
}

/**
 * @brief Another view of one row, no copy. The index is checked once.
 *
 * Row `index` starts after `index` whole rows of `columns()` values each, and is
 * `columns()` values long. Hot loops take the row once and index into the returned span,
 * so the bounds check is paid per row, not per value.
 *
 * @param index Row index, less than rows().
 * @return The `columns()` values of that row, viewing the same memory as the Tensor.
 */
std::span<const float> Tensor::row(std::size_t index) const {
    TINY_LLM_CHECK(index < rows_, std::format("row {} out of range, rows: {}", index, rows_));
    // Offset and count are in floats, relative to data_[0]: the file header was skipped by
    // whoever sliced the span, not here.
    return data_.subspan(index * columns_, columns_);
}

} // namespace tiny_llm
