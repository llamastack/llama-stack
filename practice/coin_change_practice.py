"""
Coin Change - Interview Practice Problem

Problem: You are given an integer array coins representing coins of different denominations
and an integer amount representing a total amount of money.

Return the fewest number of coins that you need to make up that amount.
If that amount of money cannot be made up by any combination of the coins, return -1.

You may assume that you have an infinite number of each kind of coin.

Example 1:
Input: coins = [1,3,4], amount = 6
Output: 2
Explanation: 6 = 3 + 3

Example 2:
Input: coins = [2], amount = 3
Output: -1

Example 3:
Input: coins = [1], amount = 0
Output: 0

Constraints:
- 1 <= coins.length <= 12
- 1 <= coins[i] <= 2^31 - 1
- 0 <= amount <= 10^4
"""


def coinChange(coins, amount):
    """
    Find the minimum number of coins needed to make the given amount.

    Args:
        coins: List[int] - denominations of coins available
        amount: int - target amount to make

    Returns:
        int - minimum number of coins needed, or -1 if impossible

    Time Complexity Goal: O(amount * len(coins))
    Space Complexity Goal: O(amount)

    Hint: This is a classic Dynamic Programming problem!
    Think about: What's the minimum coins needed for each amount from 0 to target?
    """
    min_coins = [-1 for _ in range(amount) + 1]
    min_coins[0] = 0
    for cur_amount in range(amount) + 1:
        num = [amount + 1 for _ in coins]
        for i, coin in enumerate(coins):
            hist = cur_amount - coin
            if hist >= 0 and min_coins[hist] != -1
                num[i] = min_coins[]



def test_coin_change():
    """Comprehensive test cases for coinChange function"""

    print("=" * 60)
    print("COIN CHANGE - TEST CASES")
    print("=" * 60)

    # Test Case 1: Basic example from problem description
    print("\nTest Case 1: Basic Example")
    coins1 = [1, 3, 4]
    amount1 = 6
    result1 = coinChange(coins1, amount1)
    expected1 = 2  # 3 + 3
    print(f"Coins: {coins1}, Amount: {amount1}")
    print(f"Result: {result1}, Expected: {expected1}")
    assert result1 == expected1, f"Test 1 failed: got {result1}, expected {expected1}"
    print("✅ Test 1 PASSED")

    # Test Case 2: Impossible case
    print("\nTest Case 2: Impossible Case")
    coins2 = [2]
    amount2 = 3
    result2 = coinChange(coins2, amount2)
    expected2 = -1
    print(f"Coins: {coins2}, Amount: {amount2}")
    print(f"Result: {result2}, Expected: {expected2}")
    assert result2 == expected2, f"Test 2 failed: got {result2}, expected {expected2}"
    print("✅ Test 2 PASSED")

    # Test Case 3: Zero amount
    print("\nTest Case 3: Zero Amount")
    coins3 = [1]
    amount3 = 0
    result3 = coinChange(coins3, amount3)
    expected3 = 0
    print(f"Coins: {coins3}, Amount: {amount3}")
    print(f"Result: {result3}, Expected: {expected3}")
    assert result3 == expected3, f"Test 3 failed: got {result3}, expected {expected3}"
    print("✅ Test 3 PASSED")

    # Test Case 4: Single coin solution
    print("\nTest Case 4: Single Coin Solution")
    coins4 = [1, 3, 4]
    amount4 = 4
    result4 = coinChange(coins4, amount4)
    expected4 = 1  # Just use coin 4
    print(f"Coins: {coins4}, Amount: {amount4}")
    print(f"Result: {result4}, Expected: {expected4}")
    assert result4 == expected4, f"Test 4 failed: got {result4}, expected {expected4}"
    print("✅ Test 4 PASSED")

    # Test Case 5: Greedy fails, DP succeeds
    print("\nTest Case 5: Greedy Algorithm Fails")
    coins5 = [1, 3, 4]
    amount5 = 6
    result5 = coinChange(coins5, amount5)
    expected5 = 2  # 3 + 3, not 4 + 1 + 1 (greedy would give 3)
    print(f"Coins: {coins5}, Amount: {amount5}")
    print(f"Result: {result5}, Expected: {expected5}")
    print("Note: Greedy (largest first) would give 4+1+1=3 coins, but optimal is 3+3=2 coins")
    assert result5 == expected5, f"Test 5 failed: got {result5}, expected {expected5}"
    print("✅ Test 5 PASSED")

    # Test Case 6: Large amount (performance test)
    print("\nTest Case 6: Large Amount (Performance Test)")
    coins6 = [1, 5, 10, 25]
    amount6 = 67
    result6 = coinChange(coins6, amount6)
    expected6 = 5  # 25 + 25 + 10 + 5 + 1 + 1 = 67
    print(f"Coins: {coins6}, Amount: {amount6}")
    print(f"Result: {result6}, Expected: {expected6}")
    assert result6 == expected6, f"Test 6 failed: got {result6}, expected {expected6}"
    print("✅ Test 6 PASSED")

    # Test Case 7: All ones (edge case)
    print("\nTest Case 7: Only Coin Value 1")
    coins7 = [1]
    amount7 = 5
    result7 = coinChange(coins7, amount7)
    expected7 = 5  # 1 + 1 + 1 + 1 + 1
    print(f"Coins: {coins7}, Amount: {amount7}")
    print(f"Result: {result7}, Expected: {expected7}")
    assert result7 == expected7, f"Test 7 failed: got {result7}, expected {expected7}"
    print("✅ Test 7 PASSED")

    # Test Case 8: Complex case with many coins
    print("\nTest Case 8: Complex Case")
    coins8 = [2, 3, 5]
    amount8 = 9
    result8 = coinChange(coins8, amount8)
    expected8 = 3  # 3 + 3 + 3
    print(f"Coins: {coins8}, Amount: {amount8}")
    print(f"Result: {result8}, Expected: {expected8}")
    assert result8 == expected8, f"Test 8 failed: got {result8}, expected {expected8}"
    print("✅ Test 8 PASSED")

    # Test Case 9: Impossible with multiple coins
    print("\nTest Case 9: Impossible with Multiple Coins")
    coins9 = [3, 5]
    amount9 = 1
    result9 = coinChange(coins9, amount9)
    expected9 = -1
    print(f"Coins: {coins9}, Amount: {amount9}")
    print(f"Result: {result9}, Expected: {expected9}")
    assert result9 == expected9, f"Test 9 failed: got {result9}, expected {expected9}"
    print("✅ Test 9 PASSED")

    # Test Case 10: Large performance test
    print("\nTest Case 10: Large Performance Test")
    coins10 = [1, 2, 5, 10, 20, 50]
    amount10 = 1000
    result10 = coinChange(coins10, amount10)
    expected10 = 20  # 20 coins of 50 each
    print(f"Coins: {coins10}, Amount: {amount10}")
    print(f"Result: {result10}, Expected: {expected10}")
    assert result10 == expected10, f"Test 10 failed: got {result10}, expected {expected10}"
    print("✅ Test 10 PASSED")

    print("\n" + "=" * 60)
    print("🎉 ALL TESTS PASSED! Great job!")
    print("=" * 60)


if __name__ == "__main__":
    test_coin_change()
