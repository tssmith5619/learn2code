# ==== Your function ====
def kilometers_to_miles(km):
    # 1 km = 0.621371 miles
    return km * 0.621371


# ==== Self-tests ====
assert round(kilometers_to_miles(5), 2) == 3.11
assert kilometers_to_miles(0) == 0.0

factor = 1.60934          # global variable

def miles_to_km(mi):
    factor = 1.6          # local variable, shadows the global one
    return mi * factor

print(miles_to_km(5))     # 8.0
print(factor)             # 1.60934
