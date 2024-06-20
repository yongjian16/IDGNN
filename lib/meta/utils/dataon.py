R"""
"""
#
from typing import Union, List


def dataon(
    on: Union[str, List[int]], size: int, meaningless: bool,
    /,
) -> List[int]:
    R"""
    Format data-on columns.
    """
    # Safety check.
    if meaningless and on != "none":
        # EXPECT:
        # It is possible to have improper arugments.
        raise RuntimeError(
            "Meaningless data only support data-on column identifier "
            "\"none\".",
        )

    #
    if isinstance(on, str):
        #
        if on == "all":
            #
            return list(range(size))
        elif on == "none":
            #
            return []
        else:
            # EXPECT:
            # It is possible to have improper arguments.
            raise RuntimeError(
                "Unknown data-on column identifier \"{:s}\".".format(on),
            )
    else:
        #
        return on