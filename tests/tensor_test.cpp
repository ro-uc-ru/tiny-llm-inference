#include <tiny_llm/tensor.hpp>

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <span>

namespace {

/**
 * A 3 x 4 matrix stored row after row; each value encodes its position, so a wrong index
 * shows up as a wrong number:
 *
 *     row 0:  0  1  2  3
 *     row 1: 10 11 12 13
 *     row 2: 20 21 22 23
 */
constexpr std::array<float, 12> matrix_values{0.0F,  1.0F,  2.0F,  3.0F,  10.0F, 11.0F,
                                              12.0F, 13.0F, 20.0F, 21.0F, 22.0F, 23.0F};

/** @brief The Tensor reports the shape it was built with and views all the values. */
TEST(TensorTest, KeepsShapeAndData) {
    const tiny_llm::Tensor tensor{matrix_values, {3, 4}};

    EXPECT_EQ(tensor.rows(), 3U);
    EXPECT_EQ(tensor.columns(), 4U);
    EXPECT_EQ(tensor.data().size(), 12U);
}

/** @brief Deliberate break: the buffer is allocated with new[] and never deleted. */
TEST(TensorTest, BreakOnPurposeLeaksItsBuffer) {
    float* const values = new float[12]{};
    const tiny_llm::Tensor tensor{std::span<const float>{values, 12}, {3, 4}};

    EXPECT_EQ(tensor.at(0, 0), 0.0F);
}

/** @brief A Tensor is a view: it points at the caller's memory instead of copying it. */
TEST(TensorTest, DoesNotCopyTheData) {
    const tiny_llm::Tensor tensor{matrix_values, {3, 4}};

    EXPECT_EQ(tensor.data().data(), matrix_values.data());
}

/** @brief at() reads row-major: `row * columns + column`. */
TEST(TensorTest, AtReadsRowMajor) {
    const tiny_llm::Tensor tensor{matrix_values, {3, 4}};

    EXPECT_EQ(tensor.at(0, 0), 0.0F);
    EXPECT_EQ(tensor.at(1, 2), 12.0F);
    EXPECT_EQ(tensor.at(2, 3), 23.0F); // the last valid position
}

/** @brief row() returns the right values, viewing the same memory as the Tensor. */
TEST(TensorTest, RowIsAViewOfOneRow) {
    const tiny_llm::Tensor tensor{matrix_values, {3, 4}};

    const std::span<const float> middle_row = tensor.row(1);

    ASSERT_EQ(middle_row.size(), 4U);
    EXPECT_EQ(middle_row[0], 10.0F);
    EXPECT_EQ(middle_row[3], 13.0F);
    EXPECT_EQ(middle_row.data(), matrix_values.data() + 4); // no copy: 4 floats in
}

/** @brief A vector is a single column, `{size, 1}`. */
TEST(TensorTest, VectorIsOneColumn) {
    const std::array<float, 3> vector_values{1.0F, 2.0F, 3.0F};
    const tiny_llm::Tensor vector{vector_values, {3, 1}};

    EXPECT_EQ(vector.at(2, 0), 3.0F);
    EXPECT_EQ(vector.row(1).size(), 1U);
}

// Death tests: the suite name ends in DeathTest, so GoogleTest runs them first (see
// check_test.cpp). Every message starts with the failed condition, as written in the source.

/** @brief A shape that does not match the number of values is a bug: the constructor aborts. */
TEST(TensorDeathTest, ShapeThatDoesNotMatchTheSizeAborts) {
    EXPECT_DEATH((tiny_llm::Tensor{matrix_values, {3, 5}}),
                 "rows_ \\* columns_ == data_\\.size\\(\\).*shape 3x5 needs 15 values, got 12");
}

/** @brief `row == rows()` is one past the end; the off-by-one that `>` would let through. */
TEST(TensorDeathTest, AtWithRowEqualToRowsAborts) {
    const tiny_llm::Tensor tensor{matrix_values, {3, 4}};

    EXPECT_DEATH(static_cast<void>(tensor.at(3, 0)), "row 3 out of range, rows: 3");
}

/** @brief The column has its own check, so the report says which index failed. */
TEST(TensorDeathTest, AtWithColumnEqualToColumnsAborts) {
    const tiny_llm::Tensor tensor{matrix_values, {3, 4}};

    EXPECT_DEATH(static_cast<void>(tensor.at(0, 4)), "column 4 out of range, columns: 4");
}

/** @brief row() checks its index before slicing. */
TEST(TensorDeathTest, RowEqualToRowsAborts) {
    const tiny_llm::Tensor tensor{matrix_values, {3, 4}};

    EXPECT_DEATH(static_cast<void>(tensor.row(3)), "row 3 out of range, rows: 3");
}

} // namespace
