import torch
import torch.nn as nn


class BahdanauAttention(nn.Module):

    def __init__(self, encoder_hid_size, decoder_hid_size, attn_units):
        super().__init__()
        self.attn = nn.Linear(2 * encoder_hid_size + decoder_hid_size, attn_units)
        self.V = nn.Linear(attn_units, 1)

        self.encoder_states = None
        self.enc_seq_len = None
        self.encoder_mask = None

    def get_scores(self, concated_states):
        return self.V(torch.tanh(self.attn(concated_states))).squeeze()

    def init_states(self, encoder_states):
        self.encoder_states = encoder_states
        self.encoder_mask = (encoder_states.sum(dim=2) != 0).float()
        self.enc_seq_len = encoder_states.shape[1]

    @staticmethod
    def masked_softmax(tensor, mask, dim=1):
        exps = torch.exp(tensor)
        exps = exps * mask
        divider = torch.sum(exps, keepdim=True, dim=dim)
        return exps / divider

    def calc_attention(self, dec_hidden):
        expanded_dec_hidden = dec_hidden.unsqueeze(1).repeat(1, self.enc_seq_len, 1)

        concated_states = torch.cat([self.encoder_states, expanded_dec_hidden], dim=2)
        scores = self.get_scores(concated_states)

        weights = self.masked_softmax(scores, self.encoder_mask)

        attn_vecs = (self.encoder_states * weights.unsqueeze(2)).sum(dim=1)
        return attn_vecs, weights
