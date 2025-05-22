"""Simple distance converters."""

KM_TO_MILES = 0.621371
MILES_TO_KM = 1.0 / KM_TO_MILES

def kilometers_to_miles(km: float) -> float:
    """Return km converted to miles (no rounding)."""
    return km * KM_TO_MILES

def miles_to_kilometers(mi: float) -> float:
    """Return miles converted to km (no rounding)."""
    return mi * MILES_TO_KM
