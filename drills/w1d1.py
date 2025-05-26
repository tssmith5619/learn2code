nested = [
    ["kraken_species.tsv", "dmm_clusters.csv"],
    ["sample1.fastq", "sample2.fastq", "sample3.fastq"],
]

flat = []
for inner in nested:
    flat.extend(inner)

print(flat)          # expect a single list of 5 items


orders = [
    "probiotic", "synbiotic", "prebiotic",
    "synbiotic", "probiotic", "probiotic",
]

def count_orders(items):
    counts = {}
    for name in items:
        counts[name] = counts.get(name, 0) + 1
    return counts

assert count_orders(["a", "b", "a"]) == {"a": 2, "b": 1}
assert count_orders([]) == {}
