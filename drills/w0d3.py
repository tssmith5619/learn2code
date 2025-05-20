def classify_bmi(bmi):
    if bmi < 18.5:
        return "underweight"
    elif 18.5 <= bmi < 25:
        return "normal"
    elif 25 <= bmi < 30:
        return "overweight"
    else:
        return "obese"

assert classify_bmi(22) == "normal"
assert classify_bmi(31) == "obese"

microbes = ["Bifido", "Lacto", "Bacteroides"]

for m in microbes:
    print(m.upper())

def sum_even(nums):
    total = 0
    for n in nums:
        if n % 2 != 0:          # odd → skip
            continue
        total += n
    return total

assert sum_even([1, 2, 3, 4, 5, 6]) == 12


def filter_positive_divisible(nums, divisor):
    return [n for n in nums if n > 0 and n % divisor == 0]

assert filter_positive_divisible([3, -6, 9, 12, 15], 3) == [3, 9, 12, 15]
assert filter_positive_divisible([0, 5, 10, 14], 5) == [5, 10]
