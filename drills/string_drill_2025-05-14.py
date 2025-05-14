urls = [
    "https://www.nature.com/articles/s41586-024-09999",
    "http://subdomain.nurturebio.com/orders?id=123",
    "ftp://ftp.ncbi.nlm.nih.gov/genomes/all/GCA_000001405.28",
    "https://github.com/NurtureBio/pipeline/blob/main/README.md",
]
from urllib.parse import urlparse

def extract_domain(urls):
    domains = []
    for u in urls:
        if "://" not in u:
            u = "http://" + u
        parsed = urlparse(u)
        domains.append(parsed.netloc.lstrip("www."))
    return domains
    
print(extract_domain(urls))

def test_extract_domain():
    data = [
        ("https://nurturebio.com/shop", "nurturebio.com"),
        ("ftp://ftp.ncbi.nlm.nih.gov/file.gz", "ftp.ncbi.nlm.nih.gov"),
        ("nurturebio.com/blog", "nurturebio.com"),   # no scheme
    ]
    for url, expected in data:
        assert extract_domain([url])[0] == expected, f"{url} failed"

test_extract_domain()       # should run silently now
