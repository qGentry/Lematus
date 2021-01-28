import torch
import torch.nn.functional as F


def _get_mask(target, eos_token=2):
    mask = torch.zeros_like(target).to(target.device)
    bz, maxlen = target.shape
    for b in range(bz):
        for i in range(maxlen):
            mask[b, i] = 1
            if target[b, i] == eos_token:
                break
    return mask


def calculate_seq_loss(preds, target):
    preds = preds
    target = target[:, 1:]
    assert preds.shape[:2] == target.shape
    mask = _get_mask(target)

    losses = []
    for i in range(target.shape[1]):
        loss = F.cross_entropy(preds[:, i], target[:, i], reduction='none')
        losses.append(loss)
    masked_loss = torch.stack(losses, dim=1) * mask
    return masked_loss.sum(dim=1).mean(dim=0)
