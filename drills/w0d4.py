def filter_positive_divisible(nums, divisor):
    return [n for n in nums if n > 0 and n % divisor == 0]