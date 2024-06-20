R"""
"""
#
PRECISION = 6


def floatrep(val: float) -> str:
    R"""
    Translate float number into proper string.
    """
    #
    decimal = "{:.{:d}f}".format(val, PRECISION)
    decimal = decimal.rstrip("0")
    decimal = decimal + "0" if decimal[-1] == "." else decimal

    # Base of science notation will always be a decimal floating with one
    # integer before point.
    science = "{:.{:d}e}".format(val, PRECISION)
    (base, expo) = science.split("e")
    base = base.rstrip("0")
    base = base + "0" if base[-1] == "." else base
    expo = str(int(expo))
    science = "{:s}e{:s}".format(base, expo)

    # Safety check.
    if not float(decimal) == float(science) == val:
        # EXPECT:
        # It is possible to have a float number which can not be accurately
        # expressed in given precision.
        raise RuntimeError(
            "Given float number can not be expressed accurately in precision "
            "{:d}.".format(PRECISION),
        )

    # Use shorter expression.
    # If same length, use more naive expression.
    return science if len(science) < len(decimal) else decimal