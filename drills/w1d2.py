def my_any(iterable):
    """Return True if *any* element of iterable is truthy."""
    for item in iterable:
        # TODO: add check and early return
        if item:
            return True
    return False

assert my_any([0, False, "", 5]) is True
assert my_any([0, "", None]) is False


def filter_first_n_positives(nums, n):
    result = []
    for num in nums:
        # TODO: add positivity check and quota break
        if num > 0:
            result.append(num)
        if len(result) == n:
            break
    return result


assert filter_first_n_positives([-1, 4, 0, 3, 9], 2) == [4, 3]
assert filter_first_n_positives([1, 2], 5) == [1, 2]
