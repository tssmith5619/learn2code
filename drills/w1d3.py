# drills/w1d3.py
url = "https://nurturebio.com/track?order=42&item=synbiotic&qty=3"

def query_to_dict(u):
    query = u.split("?", 1)[1]
    pairs = query.split("&")
    out = {}
    for p in pairs:
        k, v = p.split("=", 1)
        out[k] = v
    return out

print(query_to_dict(url))          # ← this must be at column 0

