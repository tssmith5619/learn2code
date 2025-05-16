def add_hmo_concentration(a, b):
    """
    Return the total milligrams of two human milk oligosaccharides (HMOs).

    Parameters
    ----------
    a : float
        Concentration of HMO 1 in mg.
    b : float
        Concentration of HMO 2 in mg.

    Returns
    -------
    float
        Total concentration in mg.
    """
    return a + b

total = add_hmo_concentration(3.4, 5.6)
print(total)       # 9.0

def clean_sequence(seq, adaptor="AGAT"):
    """
    Strip adaptor from the end of a sequencing read and return uppercase.

    Examples
    --------
    >>> clean_sequence("atcgAGAT")
    'ATCG'
    """
    seq = seq.upper()
    if seq.endswith(adaptor):
        seq = seq[:-len(adaptor)]
    return seq

assert clean_sequence("atcgAGAT") == "ATCG"
assert clean_sequence("GGTT") == "GGTT"
