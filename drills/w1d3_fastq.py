import re

FASTQ_PATTERN = re.compile(
    r"@"
    r"(?P<instrument>[A-Z0-9]+):"
    r"(?P<run_number>\d+):"
    r"(?P<flowcell_id>[A-Z0-9\-]+):"
    r"(?P<lane>\d+):"
    r"(?P<tile>\d+):"
    r"(?P<x_pos>\d+):"
    r"(?P<y_pos>\d+)\s"
    r"(?P<read>\d+):"
    r"(?P<is_filtered>[YN]):"
    r"(?P<control_number>\d+):"
    r"(?P<sample_number>\d+)"
)

def parse_fastq_header(header: str) -> dict:
    """
    Parse an Illumina-style FASTQ header into its component fields.

    Returns
    -------
    dict
        Keys: instrument, run_number, flowcell_id, lane, tile,
              x_pos, y_pos, read, is_filtered, control_number, sample_number
    """
    m = FASTQ_PATTERN.search(header)
    if not m:
        raise ValueError("Header does not match Illumina format")
    return m.groupdict()

# ---- Self-tests ----
header1 = "@M00373:49:000000000-B7JJL:1:1101:18527:1559 1:N:0:1"
header2 = "@HSQ123:7:AB12CD:2:2104:10543:678 2:Y:18:99"

assert parse_fastq_header(header1)["lane"] == "1"
assert parse_fastq_header(header2)["read"] == "2"
