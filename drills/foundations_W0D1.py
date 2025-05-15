# Scratch file: foundations_W0D1.py
shopping_list = ["milk", "eggs", "yogurt"]        # list → ordered, mutable
stock_tickers  = ("AAPL", "TSLA", "NVDA")         # tuple → ordered, immutable
ingredient_map = {"yogurt": 2, "oats": 500}       # dict  → key/value lookup
unique_species = {"Bacteroides", "Bifidobacterium"}  # set → unique items only

products = ["probiotic", "synbiotic", "probiotic", "prebiotic"]
counts = {}
for p in products:
    current = counts.get(p, 0)
    counts[p] = current + 1

print(counts)

for k, v in sorted(counts.items(), key=lambda pair: pair[1], reverse=True):
    print(f"{k}: {v}")
