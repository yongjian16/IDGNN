R"""
"""
#
import re
from typing import Dict, List, Iterable


#
INFO = Dict[str, Dict[str, str]]


def noescape(string: str, /) -> str:
    R"""
    Remove escaping charaters.
    """
    #
    return re.sub(r"\x1b\[[0-9]+(;[0-9]+)*m", "", string)


def infotab5(title: str, lines: List[str]) -> List[str]:
    R"""
    Wrap given lines into a named tab.
    """
    # Format final generated lines according to their maximum length.
    linlen = (
        0 if len(lines) == 0 else max(len(noescape(line)) for line in lines)
    )
    barlen = max(5, (linlen - len(title)) // 2)
    return (
        [
            "\x1b[2m{:s}\x1b[0m".format("-" * barlen)
            + "\x1b[1m{:s}\x1b[0m".format(" " + title + " ")
            + "\x1b[2m{:s}\x1b[0m".format("-" * barlen),
        ]
        + lines
        + ["\x1b[2m{:s}\x1b[0m".format("-" * ((barlen + 1) * 2 + len(title)))]
    )


def info5(info: INFO, /) -> str:
    R"""
    Get visible representaiton of info dict.
    """
    #
    lines = []
    for title, section in info.items():
        # Safety check.
        if any(len(key) != len(noescape(key)) for key in section.keys()):
            # UNEXPECT:
            # We can not align properly with escaping characters inside.
            raise NotImplementedError(
                "Info keys can not have escaping characters for alignment "
                "safety.",
            )

        # Ensure visible keys in the same info section have the same length.
        maxlen = max(len(key) for key in section.keys())
        buf = []
        for key, val in section.items():
            # Generate multi-line robust key-value pair representation.
            (first, *body) = val.split("\n")
            buf.append(
                "\x1b[3m{:>{:d}s}\x1b[0m".format(key, maxlen)
                + "\x1b[94m:\x1b[0m {:s}".format(first),
            )
            for line in body:
                #
                buf.append(
                    "\x1b[3m{:>{:d}s}\x1b[0m".format("", maxlen)
                    + "\x1b[90m|\x1b[0m {:s}".format(line),
                )

        # Regard an info section as a tab.
        lines.extend(infotab5(title, buf))
    return "\n".join(lines)


def infounion(info1, info2) -> INFO:
    R"""
    Merge two info dicts into a new info dict.
    """
    #
    info: INFO

    #
    info = {}
    for it in (info1, info2):
        #
        for (title, section) in it.items():
            #
            if title not in info:
                #
                info[title] = {}
            for (key, val) in section.items():
                #
                info[title][key] = val
    return info