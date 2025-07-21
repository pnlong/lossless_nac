# README
# Phillip Long
# July 20, 2025

# Serialize and Deserialize an Entropy Coder in a single byte.

# IMPORTS
##################################################

from os.path import dirname, realpath
import sys
sys.path.insert(0, dirname(realpath(__file__)))

from entropy_coder import EntropyCoder
from factory import TYPES
from verbatim import VerbatimCoder
from naive_rice import NaiveRiceCoder
from adaptive_rice import AdaptiveRiceCoder

##################################################


# CONSTANTS
##################################################

TYPE_TO_INDEX = {type_: i for i, type_ in enumerate(TYPES)} # type to index mapping
INDEX_TO_TYPE = {v: k for k, v in TYPE_TO_INDEX.items()} # index to type mapping

##################################################


# SERIALIZE
##################################################

def serialize(
    entropy_coder: EntropyCoder,
) -> int:
    """
    Serialize an entropy coder in a single byte.
    """

    # get type index
    type_ = entropy_coder.type_
    if type_ not in TYPE_TO_INDEX.keys():
        raise ValueError(f"Invalid entropy coder type: {type_}")
    type_index = TYPE_TO_INDEX[type_]

    # initialize serialized byte where the first 2 bits are the type index
    serialized = type_index << 6

    # add parameters in the remaining 6 bits
    match type_:
        case "verbatim":
            return serialized
        case "naive_rice": # for naive rice coder, we need to add the k parameter
            return serialized | entropy_coder.k
        case "adaptive_rice":
            return serialized

def deserialize(
    header_byte: int,
) -> EntropyCoder:
    """
    Deserialize an entropy coder from a single byte.
    """

    # parse the type from the first 2 bits
    type_index = header_byte >> 6
    if type_index not in TYPE_TO_INDEX.values():
        raise ValueError(f"Invalid entropy coder type index: {type_index}")
    type_ = INDEX_TO_TYPE[type_index]

    # parse the parameters from the remaining 6 bits
    match type_:
        case "verbatim":
            return VerbatimCoder()
        case "naive_rice":
            k = (header_byte & ((2 ** 6) - 1)) # last 6 bits are the k parameter
            return NaiveRiceCoder(k = k)
        case "adaptive_rice":
            return AdaptiveRiceCoder()

##################################################