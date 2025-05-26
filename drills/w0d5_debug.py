def average_positive(nums):
    """Return the average of positive numbers in nums."""
    positives = [n for n in nums if n > 0]
    if not positives:          # ← early-exit guard
        return 0.0
    return sum(positives) / len(positives)   # BUG when no positives


# ==== Self-tests ====
assert average_positive([3, 4, -2]) == 3.5
assert average_positive([-5, -1, 0]) == 0.0
