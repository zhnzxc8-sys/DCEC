import torch
import torch.nn as nn
import transformers.modeling_utils as mu
from transformers.pytorch_utils import apply_chunking_to_forward

mu.apply_chunking_to_forward = apply_chunking_to_forward

def find_pruneable_heads_and_indices(heads, n_heads, head_size, already_pruned_heads):
    mask = torch.ones(n_heads, head_size)
    heads = set(heads) - already_pruned_heads

    for head in heads:
        head = head - sum(1 for h in already_pruned_heads if h < head)
        mask[head] = 0

    mask = mask.view(-1).contiguous().eq(1)
    index = torch.arange(len(mask))[mask].long()
    return heads, index

def prune_linear_layer(layer, index, dim=0):
    index = index.to(layer.weight.device)

    W = layer.weight.index_select(dim, index).clone().detach()

    if layer.bias is not None:
        if dim == 1:
            b = layer.bias.clone().detach()
        else:
            b = layer.bias[index].clone().detach()

    new_size = list(layer.weight.size())
    new_size[dim] = len(index)

    new_layer = nn.Linear(
        new_size[1],
        new_size[0],
        bias=layer.bias is not None
    ).to(layer.weight.device)

    new_layer.weight.data.copy_(W.contiguous())

    if layer.bias is not None:
        new_layer.bias.data.copy_(b.contiguous())

    return new_layer

mu.find_pruneable_heads_and_indices = find_pruneable_heads_and_indices
mu.prune_linear_layer = prune_linear_layer
