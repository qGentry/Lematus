def _unpad_seq(batch_inds, eos_token=2):
    result = []
    cur_seq = []
    for point in batch_inds:
        for ind in point:
            if ind != eos_token:
                cur_seq.append(ind.item())
            else:
                break
        result.append(cur_seq)
        cur_seq = []
    return result


def calc_accuracy(preds, target):
    pred_inds = preds.argmax(dim=2)
    target = target[:, 1:]
    pred_seqs = _unpad_seq(pred_inds)
    true_seqs = _unpad_seq(target)
    goods = 0
    for y_pred, y_true in zip(pred_seqs, true_seqs):
        if y_pred == y_true:
            goods += 1
    return goods / pred_inds.shape[0]
