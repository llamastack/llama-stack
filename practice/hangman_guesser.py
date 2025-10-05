from collections import defaultdict


def guess_next_character(mystery_word_pattern, guessed_characters, word_pool):
    """
    Returns:
        str: Single character that is most likely to be in the mystery word,
             or None if no good guess can be made
    """
    num_chars = len(mystery_word_pattern)
    correct_positions = {i: c for i, c in enumerate(mystery_word_pattern) if c != "_"}
    char_counts_by_position = [defaultdict(int) for _ in range(num_chars)]

    for word in word_pool:
        wordlen = len(word)
        matches = True
        for i, c in correct_positions.items():
            if i < wordlen and word[i] != c:
                matches = False
                break
        if not matches:
            continue

        for j, c in enumerate(word):
            if j >= num_chars:
                continue
            if c in guessed_characters:
                continue
            if mystery_word_pattern[j] != "_":
                continue
            char_counts_by_position[j][c] += 1

    max_count = 0
    char = None
    for counts in char_counts_by_position:
        for c, v in counts.items():
            if v > max_count:
                max_count = v
                char = c
    return char


# Test cases
def test_guess_next_character():
    """Test cases for the hangman character guesser"""

    # Test case 1: Basic case with clear winner
    pattern = "_at"
    guessed = ["a", "t"]
    pool = ["cat", "bat", "hat", "rat", "mat"]
    result = guess_next_character(pattern, guessed, pool)
    # Any of 'c', 'b', 'h', 'r', 'm' would be valid since they all appear once
    assert result in ["c", "b", "h", "r", "m"], f"Expected one of 'c','b','h','r','m', got {result}"

    # Test case 2: Some characters already guessed
    pattern = "_at"
    guessed = ["a", "t", "c", "b"]
    pool = ["cat", "bat", "hat", "rat", "mat"]
    result = guess_next_character(pattern, guessed, pool)
    assert result in ["h", "r", "m"], f"Expected one of 'h','r','m', got {result}"

    # Test case 3: Multiple missing positions
    pattern = "_a_e"
    guessed = ["a", "e"]
    pool = ["cake", "bake", "lake", "make", "take", "wake", "came", "name", "game", "same"]
    result = guess_next_character(pattern, guessed, pool)
    # Should return most frequent character in missing positions
    assert isinstance(result, str) and len(result) == 1, f"Expected single character, got {result}"

    # Test case 4: Word pool doesn't match pattern (should filter)
    pattern = "_at"
    guessed = ["a", "t"]
    pool = ["cat", "dog", "bat", "rat", "hat"]  # "dog" doesn't match the pattern
    result = guess_next_character(pattern, guessed, pool)
    assert result in ["c", "b", "r", "h"], f"Expected one of 'c','b','r','h', got {result}"

    # Test case 5: All characters in matching words already guessed
    pattern = "_at"
    guessed = ["a", "t", "c", "b", "h", "r", "m"]
    pool = ["cat", "bat", "hat", "rat", "mat"]
    result = guess_next_character(pattern, guessed, pool)
    assert result is None, f"Expected None when all characters guessed, got {result}"

    # Test case 6: Single character missing
    pattern = "c_t"
    guessed = ["c", "t"]
    pool = ["cat", "cot", "cut", "cit"]
    result = guess_next_character(pattern, guessed, pool)
    # Should return most frequent vowel/character in position 1
    assert isinstance(result, str) and len(result) == 1, f"Expected single character, got {result}"

    # Test case 7: Empty word pool
    pattern = "_at"
    guessed = ["a", "t"]
    pool = []
    result = guess_next_character(pattern, guessed, pool)
    assert result is None, f"Expected None for empty pool, got {result}"

    # Test case 8: No matching words in pool
    pattern = "_at"
    guessed = ["a", "t"]
    pool = ["dog", "run", "sun"]
    result = guess_next_character(pattern, guessed, pool)
    assert result is None, f"Expected None when no words match pattern, got {result}"

    # Test case 9: Longer word with multiple gaps
    pattern = "_o_er"
    guessed = ["o", "e", "r"]
    pool = ["power", "tower", "lower", "cover", "hover", "mower", "boxer", "poker"]
    result = guess_next_character(pattern, guessed, pool)
    assert isinstance(result, str) and len(result) == 1, f"Expected single character, got {result}"

    # Test case 10: Case sensitivity
    pattern = "_at"
    guessed = ["a", "t"]
    pool = ["Cat", "BAT", "hat"]  # Mixed case
    result = guess_next_character(pattern, guessed, pool)
    # Should handle case appropriately
    assert result is not None, f"Expected a character, got {result}"

    # Test case 11: Tie-breaking - multiple characters with same frequency
    pattern = "_a_"
    guessed = ["a"]
    pool = ["cat", "bat", "had", "bag"]  # c,b,h,g all appear once in pos 0; t,t,d,g in pos 2
    result = guess_next_character(pattern, guessed, pool)
    # Should return one of the valid characters (implementation dependent)
    assert result is not None and result not in guessed, f"Expected unguessed character, got {result}"

    # Test case 12: Complex pattern with repeated characters in word
    pattern = "_oo_"
    guessed = ["o"]
    pool = ["book", "look", "took", "cook", "hook", "noon", "boom", "doom"]
    result = guess_next_character(pattern, guessed, pool)
    assert result is not None and result != "o", f"Expected character other than 'o', got {result}"

    # Test case 13: Very long word with sparse information
    pattern = "___e____i__"
    guessed = ["e", "i"]
    pool = ["programming", "engineering", "mathematics", "development"]
    result = guess_next_character(pattern, guessed, pool)
    # Only "programming" and "engineering" match the pattern
    assert result is not None, f"Expected a character, got {result}"

    # Test case 14: Word with all same length but different patterns
    pattern = "_a__a"
    guessed = ["a"]
    pool = ["mamma", "drama", "llama", "karma", "panda"]  # "panda" doesn't match
    result = guess_next_character(pattern, guessed, pool)
    # Should only consider mamma, drama, llama, karma
    assert result is not None and result != "a", f"Expected character other than 'a', got {result}"

    # Test case 15: Single letter word
    pattern = "_"
    guessed = []
    pool = ["a", "I", "o"]
    result = guess_next_character(pattern, guessed, pool)
    assert result in ["a", "I", "o"], f"Expected one of 'a','I','o', got {result}"

    # Test case 16: Almost complete word - only one missing
    pattern = "almos_"
    guessed = ["a", "l", "m", "o", "s"]
    pool = ["almost", "almond"]  # "almond" doesn't match pattern
    result = guess_next_character(pattern, guessed, pool)
    assert result == "t", f"Expected 't', got {result}"

    # Test case 17: Frequency analysis with position weighting
    pattern = "_e__e_"
    guessed = ["e"]
    pool = ["better", "letter", "pepper", "keeper", "helper", "member"]
    result = guess_next_character(pattern, guessed, pool)
    # Should find most frequent character across all missing positions
    assert result is not None and result != "e", f"Expected character other than 'e', got {result}"

    # Test case 18: Words with different lengths in pool (should filter by length)
    pattern = "___"
    guessed = []
    pool = ["cat", "dog", "fox", "car", "bar", "bat", "run"]
    result = guess_next_character(pattern, guessed, pool)
    # Should only consider 3-letter words: cat, dog, fox, car, bar, bat, run
    assert result is not None, f"Expected a character, got {result}"

    # Test case 19: All positions filled except one, but multiple valid completions
    pattern = "c_r"
    guessed = ["c", "r"]
    pool = ["car", "cor", "cur", "cir"]  # All valid completions
    result = guess_next_character(pattern, guessed, pool)
    assert result in ["a", "o", "u", "i"], f"Expected one of 'a','o','u','i', got {result}"

    # Test case 20: Pattern with numbers/special chars (edge case)
    pattern = "_a_"
    guessed = ["a"]
    pool = ["1a2", "3a4", "cat", "bat"]  # Mix of alphanumeric and letters
    result = guess_next_character(pattern, guessed, pool)
    # Should handle all valid characters
    assert result is not None, f"Expected a character, got {result}"

    # Test case 21: Very large frequency difference
    pattern = "_a_"
    guessed = ["a"]
    pool = ["cat"] * 100 + ["bat", "hat", "rat"]  # 'c' appears 100 times, others once each
    result = guess_next_character(pattern, guessed, pool)
    assert result == "c", f"Expected 'c' due to high frequency, got {result}"

    # Test case 22: Pattern where some positions have no valid characters
    pattern = "_x_"
    guessed = [
        "x",
        "a",
        "b",
        "c",
        "d",
        "e",
        "f",
        "g",
        "h",
        "i",
        "j",
        "k",
        "l",
        "m",
        "n",
        "o",
        "p",
        "q",
        "r",
        "s",
        "t",
        "u",
        "v",
        "w",
        "y",
        "z",
    ]
    pool = ["axe", "fox", "box", "six"]
    result = guess_next_character(pattern, guessed, pool)
    # All common letters guessed, should return None or a less common letter
    assert result is None or result not in guessed, f"Unexpected result: {result}"

    print("All test cases passed!")


if __name__ == "__main__":
    test_guess_next_character()
