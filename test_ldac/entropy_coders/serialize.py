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
from partitioned_rice import PartitionedRiceCoder

##################################################


# CONSTANTS
##################################################

TYPE_TO_INDEX = {type_: i for i, type_ in enumerate(TYPES)} # type to index mapping
INDEX_TO_TYPE = {v: k for k, v in TYPE_TO_INDEX.items()} # index to type mapping
SERIALIZED_ENTROPY_CODER_BITS = 8 # 8 bits for the serialized entropy coder
N_BITS_FOR_TYPE = 2 # number of bits for the type
N_BITS_FOR_PARAMETERS = SERIALIZED_ENTROPY_CODER_BITS - N_BITS_FOR_TYPE # number of bits for the parameters

##################################################


# HELPER FUNCTIONS
##################################################

def get_parameters_from_header(
    header: int,
) -> int:
    """
    Get the parameters from a header byte.

    Parameters
    ----------
    header : int
        The header byte to get the parameters from.

    Returns
    -------
    int
        The parameters.
    """
    return header & ((2 ** N_BITS_FOR_PARAMETERS) - 1) # last N_BITS_FOR_PARAMETERS bits are the parameters

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
    serialized = type_index << N_BITS_FOR_PARAMETERS

    # add parameters in the remaining 6 bits
    match type_:
        case "verbatim":
            return serialized
        case "naive_rice": # for naive rice coder, we need to add the k parameter
            return serialized | entropy_coder.k
        case "adaptive_rice":
            return serialized
        case "partitioned_rice":
            return serialized | entropy_coder.partition_size_factor

def deserialize(
    header: int,
) -> EntropyCoder:
    """
    Deserialize an entropy coder from a single byte.
    """

    # parse the type from the first 2 bits
    type_index = header >> N_BITS_FOR_PARAMETERS
    if type_index not in TYPE_TO_INDEX.values():
        raise ValueError(f"Invalid entropy coder type index: {type_index}")
    type_ = INDEX_TO_TYPE[type_index]

    # parse the parameters from the remaining 6 bits
    match type_:
        case "verbatim":
            return VerbatimCoder()
        case "naive_rice":
            k = get_parameters_from_header(header = header)
            return NaiveRiceCoder(k = k)
        case "adaptive_rice":
            return AdaptiveRiceCoder()
        case "partitioned_rice":
            partition_size_factor = get_parameters_from_header(header = header)
            return PartitionedRiceCoder(partition_size_factor = partition_size_factor)

##################################################