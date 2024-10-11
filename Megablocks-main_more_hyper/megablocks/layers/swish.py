import stk
import torch
import torch.nn.functional as F


def swish(x: stk.Matrix):
    assert isinstance(x, stk.Matrix)
    return stk.Matrix(
        x.size(),
        F.silu(x.data),
        x.row_indices,
        x.column_indices,
        x.offsets,
        x.column_indices_t,
        x.offsets_t,
        x.block_offsets_t)
