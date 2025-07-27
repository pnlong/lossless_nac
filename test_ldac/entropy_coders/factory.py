# README
# Phillip Long
# July 20, 2025

# IMPORTS
##################################################

from os.path import dirname, realpath
import sys
sys.path.insert(0, dirname(realpath(__file__)))

from entropy_coder import EntropyCoder
from verbatim import VerbatimCoder
from naive_rice import NaiveRiceCoder
from adaptive_rice import AdaptiveRiceCoder
from partitioned_rice import PartitionedRiceCoder

##################################################


# CONSTANTS
##################################################

TYPES = ["verbatim", "naive_rice", "adaptive_rice", "partitioned_rice"]

##################################################


# FACTORY
##################################################

def get_entropy_coder(
    type_: str,
    **kwargs,
) -> EntropyCoder:
    """
    Get an entropy coder of the given type.

    Parameters
    ----------
    type_ : str
        The type of entropy coder to get.
    **kwargs : dict
        Additional keyword arguments to pass to the entropy coder constructor.

    Returns
    -------
    EntropyCoder
        The entropy coder.
    """

    match type_:
        case "verbatim":
            return VerbatimCoder(**kwargs)
        case "naive_rice":
            return NaiveRiceCoder(**kwargs)
        case "adaptive_rice":
            return AdaptiveRiceCoder(**kwargs)
        case "partitioned_rice":
            return PartitionedRiceCoder(**kwargs)
        case _:
            raise ValueError(f"Invalid entropy coder type: {type_}")

##################################################