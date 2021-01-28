import torch
import torch.nn as nn
from lematus.model.layers.attention import BahdanauAttention


class DecoderNetwork(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device

    @staticmethod
    def keep_decoding(train, stopped_flag_tensor):
        if train:
            return True
        else:
            return not torch.all(stopped_flag_tensor)

    def get_final_inds_tensor(self, hidden):
        bz = hidden.shape[0]
        return (torch.ones([bz], dtype=torch.long) * -1).to(self.device)

    def get_stopped_flag_tensor(self, hidden):
        bz = hidden.shape[0]
        return torch.tensor([False] * bz).to(self.device)

    def get_bos_tokens(self, hidden):
        bz = hidden.shape[0]
        return self.bos_token * torch.ones([bz], dtype=torch.long).to(self.device)


class ConditionalDecoder(DecoderNetwork):

    def __init__(self,
                 enc_hid_dim: int,
                 dec_hid_dim: int,
                 emb_dim: int,
                 emb_count: int,
                 attn_dim: int,
                 dropout: float = 0.2,
                 teacher_forcing_rate: float = 0.0,
                 maxlen: int = 30,
                 eos_token: int = 2,
                 bos_token: int = 1,
                 device='cpu',
                 ):
        super().__init__(device)
        self.embeddings = nn.Embedding(
            embedding_dim=emb_dim,
            num_embeddings=emb_count,
            padding_idx=0,
        )

        self.bos_token = bos_token
        self.eos_token = eos_token
        self.maxlen = maxlen

        self.rnn_cell_bottom = nn.GRUCell(emb_dim, dec_hid_dim)
        self.attn_module = BahdanauAttention(enc_hid_dim, dec_hid_dim, attn_dim)
        self.rnn_cell_top = nn.GRUCell(dec_hid_dim, enc_hid_dim * 2)

        self.recurrent_fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim)

        self.dropout = nn.Dropout(dropout)

        self.teacher_forcing_rate = teacher_forcing_rate
        self.output_proj = nn.Linear(dec_hid_dim, emb_count)

        self.recurrent_fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim)

        self.relu = nn.ReLU()

    def process_input(self, inputs, state):
        embedded = self.embeddings(inputs)
        step1 = self.rnn_cell_bottom(embedded, state)
        step1 = self.dropout(step1)

        attn_vec, weights = self.attn_module.calc_attention(step1)
        attn_vec = self.dropout(attn_vec)

        step2 = self.rnn_cell_top(step1, attn_vec)

        step2 = self.relu(step2)
        step2 = self.recurrent_fc(step2)

        return step2, weights

    def decode(self, hidden, enc_states, true_labels=None, train=False):
        assert (train and true_labels is not None) or (not train and true_labels is None)
        self.attn_module.init_states(enc_states)

        inputs = self.get_bos_tokens(hidden)

        state, weights = self.process_input(inputs, hidden)
        attn_weights = [weights]
        states = [state]

        final_inds_tensor = self.get_final_inds_tensor(hidden)
        stopped_flag_tensor = self.get_stopped_flag_tensor(hidden)

        steps = 1

        if train:
            maxlen = true_labels.shape[1] - 1
        else:
            maxlen = self.maxlen
        while steps < maxlen and self.keep_decoding(train, stopped_flag_tensor):
            steps += 1
            if train and torch.rand(1) < self.teacher_forcing_rate:
                inputs = true_labels[:, steps]
            else:
                output_proj = self.output_proj(state)
                inputs = torch.argmax(output_proj, dim=1)
            state, weights = self.process_input(inputs, state)
            attn_weights.append(weights)
            states.append(state)

            stopped_flag_tensor = stopped_flag_tensor | (inputs == self.eos_token)
            final_inds_tensor = torch.where(
                stopped_flag_tensor & (final_inds_tensor == -1),
                torch.tensor(steps).to(self.device),
                final_inds_tensor,
            )
        final_inds_tensor = torch.where(final_inds_tensor == -1,
                                        torch.tensor(steps).to(self.device),
                                        final_inds_tensor)

        states = torch.stack(states, dim=1)
        attn_weights = torch.stack(attn_weights, dim=1)
        preds = self.output_proj(states)
        return preds, final_inds_tensor, attn_weights

