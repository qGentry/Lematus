import torch
import torch.nn as nn


class EncoderNetwork(nn.Module):

    def __init__(self,
                 enc_hid_dim: int,
                 emb_dim: int,
                 dec_hid_dim: int,
                 emb_count: int,
                 dropout: float,
                 num_layers: int,
                 device: str = 'cpu'):
        self.device = device
        super().__init__()
        self.embeddings = nn.Embedding(
            num_embeddings=emb_count,
            embedding_dim=emb_dim,
            padding_idx=0,
        )

        self.rnn = nn.GRU(
            input_size=emb_dim,
            hidden_size=enc_hid_dim,
            dropout=dropout,
            bidirectional=True,
            batch_first=True,
            num_layers=num_layers,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(2 * num_layers * enc_hid_dim, dec_hid_dim)

    def get_initial_state(self, inp):
        shape = self.rnn.get_expected_hidden_size(inp, None)
        return torch.zeros(shape).to(self.device)

    def forward(self, x):
        x = x.to(self.device)
        lens = (x != 0).sum(dim=1)
        x = self.embeddings(x)
        packed = torch.nn.utils.rnn.pack_padded_sequence(
            x,
            lengths=lens,
            batch_first=True,
            enforce_sorted=False
        )
        states, last_hidden = self.rnn(packed, self.get_initial_state(x))
        states, lens = torch.nn.utils.rnn.pad_packed_sequence(states, batch_first=True)
        last_hidden = torch.cat([*last_hidden], dim=1)

        last_hidden = self.fc(last_hidden)
        last_hidden = torch.tanh(last_hidden)
        last_hidden = self.dropout(last_hidden)

        return (states, lens), last_hidden
