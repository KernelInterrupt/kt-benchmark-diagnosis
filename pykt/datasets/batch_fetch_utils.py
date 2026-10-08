import numpy as np
import torch


def batch_select(value, batch_indices):
    if torch.is_tensor(value):
        if torch.is_tensor(batch_indices):
            index = batch_indices.to(device=value.device, dtype=torch.long)
        else:
            index = torch.as_tensor(list(batch_indices), device=value.device, dtype=torch.long)
        return value.index_select(0, index)
    if isinstance(value, np.ndarray):
        if torch.is_tensor(batch_indices):
            batch_indices = batch_indices.detach().cpu().tolist()
        return value[batch_indices]
    if isinstance(value, list):
        if torch.is_tensor(batch_indices):
            batch_indices = batch_indices.detach().cpu().tolist()
        return [value[int(i)] for i in batch_indices]
    return value


def apply_sequence_mask(batch_value, masks):
    cur_masks = masks
    while torch.is_tensor(cur_masks) and cur_masks.ndim < batch_value.ndim:
        cur_masks = cur_masks.unsqueeze(-1)
    return batch_value * cur_masks


def shifted_masked_batch(batch_value, masks, apply_mask=True):
    seqs = batch_value[:, :-1, ...]
    shft = batch_value[:, 1:, ...]
    if apply_mask:
        seqs = apply_sequence_mask(seqs, masks)
        shft = apply_sequence_mask(shft, masks)
    return seqs, shft


def expand_empty_batch(value, batch_size):
    if torch.is_tensor(value):
        view = value.unsqueeze(0)
        repeats = [int(batch_size)] + [1] * value.ndim
        return view.repeat(*repeats)
    if isinstance(value, np.ndarray):
        return np.repeat(value[np.newaxis, ...], int(batch_size), axis=0)
    if isinstance(value, list):
        return [list(value) for _ in range(int(batch_size))]
    return value
