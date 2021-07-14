from typing import Sequence

from swyft.types import MarginalsType, StrictMarginalsType


def _corner_params(params):
    out = []
    for i in range(len(params)):
        for j in range(i, len(params)):
            if i == j:
                out.append((params[i],))
            else:
                out.append((params[i], params[j]))
    return out


# def sort_param_list(param_list: Sequence):
#    result = []
#    for v in param_list:
#        if not isinstance(v, tuple):
#            v = (v,)
#        else:
#            v = tuple(sorted(v))
#        result.append(v)
#    return result


# def format_param_list(params, all_params=None, mode="custom"):
#    # Use all parameters if params == None
#    if params is None and all_params is None:
#        raise ValueError("Specify parameters!")
#    if params is None:
#        params = all_params
#
#    if mode == "custom" or mode == "1d":
#        param_list = params
#    elif mode == "2d":
#        param_list = _corner_params(params)
#    else:
#        raise KeyError("Invalid mode argument.")
#
#    return sort_param_list(param_list)


def tupleize_marginals(marginals: MarginalsType) -> StrictMarginalsType:
    """Reformat input marginals into sorted and hashable standard form: tuples of tuples"""
    out = list(marginals)
    for i in range(len(out)):
        if isinstance(out[i], int):
            out[i] = (out[i],)
        else:
            out[i] = tuple(sorted(set(out[i])))
    out = tuple(sorted(out))
    return out
