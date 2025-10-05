"""
Rotate Image (Matrix) - Interview Practice Problem

Problem: Given an n x n 2D matrix representing an image, rotate it by 90 degrees clockwise in-place.
You have to rotate the image in-place, which means you have to modify the input 2D matrix directly.
DO NOT allocate another 2D matrix and do the rotation.

Example:
Input:  [[1,2,3],
         [4,5,6],
         [7,8,9]]

Output: [[7,4,1],
         [8,5,2],
         [9,6,3]]

Constraints:
- n == matrix.length == matrix[i].length
- 1 <= n <= 20
- -1000 <= matrix[i][j] <= 1000
"""


def rotate(matrix):
    """
    Rotate the matrix 90 degrees clockwise in-place.

    Args:
        matrix: List[List[int]] - n x n 2D matrix

    Returns:
        None - Do not return anything, modify matrix in-place instead.

    Time Complexity Goal: O(n^2)
    Space Complexity Goal: O(1)
    """
    a = matrix
    n = len(a[0])

    # first swap against one diagonal so a(i, j) -> a(n - j - 1, n - i - 1)
    for i in range(0, n):
        for j in range(0, n - i):
            t = a[i][j]
            a[i][j] = a[n - j - 1][n - i - 1]
            a[n - j - 1][n - i - 1] = t

    # now flip across horizontal line
    for i in range(0, n // 2):
        for j in range(0, n):
            t = a[i][j]
            a[i][j] = a[n - i - 1][j]
            a[n - i - 1][j] = t


def print_matrix(matrix, title="Matrix"):
    """Helper function to print matrix in a readable format"""
    print(f"\n{title}:")
    for row in matrix:
        print(row)


def test_rotate():
    """Comprehensive test cases for rotate function"""

    print("=" * 60)
    print("ROTATE IMAGE - TEST CASES")
    print("=" * 60)

    # Test Case 1: 3x3 matrix (basic example)
    print("\nTest Case 1: 3x3 Matrix")
    matrix1 = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    expected1 = [[7, 4, 1], [8, 5, 2], [9, 6, 3]]
    print_matrix(matrix1, "Before rotation")
    rotate(matrix1)
    print_matrix(matrix1, "After rotation")
    print_matrix(expected1, "Expected")
    assert matrix1 == expected1, f"Test 1 failed: got {matrix1}, expected {expected1}"
    print("✅ Test 1 PASSED")

    # Test Case 2: 1x1 matrix (edge case)
    print("\nTest Case 2: 1x1 Matrix (Edge Case)")
    matrix2 = [[42]]
    expected2 = [[42]]
    print_matrix(matrix2, "Before rotation")
    rotate(matrix2)
    print_matrix(matrix2, "After rotation")
    assert matrix2 == expected2, f"Test 2 failed: got {matrix2}, expected {expected2}"
    print("✅ Test 2 PASSED")

    # Test Case 3: 2x2 matrix
    print("\nTest Case 3: 2x2 Matrix")
    matrix3 = [[1, 2], [3, 4]]
    expected3 = [[3, 1], [4, 2]]
    print_matrix(matrix3, "Before rotation")
    rotate(matrix3)
    print_matrix(matrix3, "After rotation")
    print_matrix(expected3, "Expected")
    assert matrix3 == expected3, f"Test 3 failed: got {matrix3}, expected {expected3}"
    print("✅ Test 3 PASSED")

    # Test Case 4: 4x4 matrix (larger matrix)
    print("\nTest Case 4: 4x4 Matrix")
    matrix4 = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]
    expected4 = [[13, 9, 5, 1], [14, 10, 6, 2], [15, 11, 7, 3], [16, 12, 8, 4]]
    print_matrix(matrix4, "Before rotation")
    rotate(matrix4)
    print_matrix(matrix4, "After rotation")
    print_matrix(expected4, "Expected")
    assert matrix4 == expected4, f"Test 4 failed: got {matrix4}, expected {expected4}"
    print("✅ Test 4 PASSED")

    # Test Case 5: Matrix with negative numbers
    print("\nTest Case 5: Matrix with Negative Numbers")
    matrix5 = [[-1, -2, -3], [-4, -5, -6], [-7, -8, -9]]
    expected5 = [[-7, -4, -1], [-8, -5, -2], [-9, -6, -3]]
    print_matrix(matrix5, "Before rotation")
    rotate(matrix5)
    print_matrix(matrix5, "After rotation")
    print_matrix(expected5, "Expected")
    assert matrix5 == expected5, f"Test 5 failed: got {matrix5}, expected {expected5}"
    print("✅ Test 5 PASSED")

    # Test Case 6: 5x5 matrix (odd dimension, complexity test)
    print("\nTest Case 6: 5x5 Matrix (Complexity Test)")
    matrix6 = [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15], [16, 17, 18, 19, 20], [21, 22, 23, 24, 25]]
    expected6 = [[21, 16, 11, 6, 1], [22, 17, 12, 7, 2], [23, 18, 13, 8, 3], [24, 19, 14, 9, 4], [25, 20, 15, 10, 5]]
    print_matrix(matrix6, "Before rotation")
    rotate(matrix6)
    print_matrix(matrix6, "After rotation")
    print_matrix(expected6, "Expected")
    assert matrix6 == expected6, f"Test 6 failed: got {matrix6}, expected {expected6}"
    print("✅ Test 6 PASSED")

    print("\n" + "=" * 60)
    print("🎉 ALL TESTS PASSED! Great job!")
    print("=" * 60)


if __name__ == "__main__":
    test_rotate()
