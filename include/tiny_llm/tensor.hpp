#pragma once

#include <array>
#include <cstddef>
#include <span>

namespace tiny_llm {

/**
 * @brief A read-only, row-major 2-D view of `float` data that someone else owns.
 *
 * A Tensor owns nothing: it is a pointer, a length and a shape (32 bytes). The weights
 * belong to the MappedFile they were loaded from, so no Tensor may outlive it; nothing in
 * C++ enforces that, the owner's lifetime does.
 *
 * Invariant: `rows() * columns() == data().size()`, checked once in the constructor.
 * The members are private so that the shape cannot change after that check.
 *
 * A vector is stored as a single column, `{size, 1}`. There are no strides and no dtype:
 * the legacy llama2.c format is float32 only, and the conversion from file bytes to floats
 * happens in the loader, not here. `std::mdspan` is not used: libstdc++ only ships it from
 * GCC 15, and without strides it would add nothing over `row * columns + column`.
 */
class Tensor {
  public:
    /**
     * @brief Wraps `data` as a `rows x columns` matrix, without copying.
     *
     * A mismatch between the shape and the size is a bug in the code that sliced the
     * weights (a truncated file is rejected earlier, with LoadError), so it aborts with
     * TINY_LLM_CHECK.
     *
     * @param data The values, row after row. Must outlive the Tensor.
     * @param shape `{rows, columns}`.
     */
    Tensor(std::span<const float> data, std::array<std::size_t, 2> shape);

    /** @return The number of rows. */
    [[nodiscard]] std::size_t rows() const noexcept { return rows_; }

    /** @return The number of columns. */
    [[nodiscard]] std::size_t columns() const noexcept { return columns_; }

    /** @return The whole view, `rows() * columns()` values. */
    [[nodiscard]] std::span<const float> data() const noexcept { return data_; }

    /**
     * @brief One value, with both indices checked.
     *
     * For tests and cold code; hot loops take a row() once and index into it.
     *
     * @param row Row index, less than rows().
     * @param column Column index, less than columns().
     * @return The value at `data[row * columns() + column]`.
     */
    [[nodiscard]] float at(std::size_t row, std::size_t column) const;

    /**
     * @brief Another view of one row, no copy. The index is checked once.
     *
     * @param index Row index, less than rows().
     * @return The `columns()` values of that row.
     */
    [[nodiscard]] std::span<const float> row(std::size_t index) const;

  private:
    std::span<const float> data_; /**< The values; owned by someone else. */
    std::size_t rows_;            /**< Number of rows. */
    std::size_t columns_;         /**< Number of columns. */
};

} // namespace tiny_llm
