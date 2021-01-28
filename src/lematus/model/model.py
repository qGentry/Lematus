import torch
import torch.nn as nn
import os
import yaml

from lematus.data_utils.preprocessing import Preprocessor
from lematus.model.layers.decoder import ConditionalDecoder
from lematus.model.layers.encoder import EncoderNetwork


class ConditionalSeq2SeqModel(nn.Module):

    def __init__(self, encoder_params, decoder_params, input_token2idx, target_token2idx, device='cpu'):
        super().__init__()
        self.device = device

        self.input_token2idx = input_token2idx
        self.input_idx2token = {i: char for char, i in self.input_token2idx.items()}

        self.target_token2idx = target_token2idx
        self.target_idx2token = {i: char for char, i in self.target_token2idx.items()}

        self.encoder = EncoderNetwork(emb_count=len(input_token2idx), **encoder_params)
        self.decoder = ConditionalDecoder(emb_count=len(target_token2idx), **decoder_params)

        self.to(device)

    @classmethod
    def load_saved_model(cls, model_config, path, prefix=''):
        with open(os.path.join(path, "input_token2idx.yaml"), 'w') as f:
            input_idx2token = yaml.dump(f)
        with open(os.path.join(path, "target_token2idx"), 'w') as f:
            target_token2idx = yaml.dump(f)

        model = cls(model_config['encoder_params'], model_config['decoder_params'], input_idx2token, target_token2idx)

        model_name = prefix + "_" + "model_state_dict"
        state_dict = torch.load(os.path.join(path, model_name))
        model.load_state_dict(state_dict)

        return model

    def save_model(self, path, prefix=''):
        model_name = prefix + "_" + "model_state_dict"
        torch.save(self.state_dict(), os.path.join(path, model_name))

        with open(os.path.join(path, "input_token2idx.yaml"), 'w') as f:
            yaml.dump(self.input_idx2token, f)
        with open(os.path.join(path, "target_token2idx.yaml"), 'w') as f:
            yaml.dump(self.target_token2idx, f)

    def forward(self, x, true_labels=None, train=False):
        x = x.to(self.device)
        if true_labels is not None:
            true_labels = true_labels.to(self.device)
        states, last_hidden = self.encoder(x)
        preds, lens, attn_weights = self.decoder.decode(last_hidden, states[0], true_labels, train)
        return preds, lens, attn_weights

    def lemmatize_text(self, text):
        text, lc, rc = self.prepare_for_lemmatization(text)
        inp_list = self.prepare_inference_input(text, lc, rc)
        inp_tensor = torch.LongTensor(inp_list).unsqueeze(0)
        pred, _, _ = self.forward(inp_tensor)
        pred = pred.squeeze()
        length = pred.shape[0]
        inds = []

        for i in range(length):
            cur_pred = torch.argmax(pred[i])
            inds.append(cur_pred.item())
        return self.stringify(inds, mode="target")

    def stringify(self, inds, mode):
        if mode == "input":
            idx2token = self.input_idx2token
        elif mode == "target":
            idx2token = self.target_idx2token
        else:
            raise ValueError
        chars = []
        for ind in inds:
            chars.append(idx2token[ind])
            if ind == 2:
                break
        return ''.join(chars)

    def prepare_inference_input(self, text, lc, rc):
        text = Preprocessor.preprocess(text)
        lc = Preprocessor.preprocess(lc)
        rc = Preprocessor.preprocess(rc)

        text_tokenized = [self.input_token2idx[char] for char in text]

        lc_tokenized = [self.input_token2idx[char] for char in lc]
        lc_border = [self.input_token2idx["<lc>"]]

        rc_tokenized = [self.input_token2idx[char] for char in rc]
        rc_border = [self.input_token2idx["<rc>"]]

        return lc_tokenized + lc_border + text_tokenized + rc_border + rc_tokenized

    @staticmethod
    def prepare_for_lemmatization(text, window_size=25, sep="|"):
        lc, target, rc = text.split(sep)
        if len(lc) > window_size:
            lc = lc[-window_size:]
        if len(rc) > window_size:
            rc = rc[:window_size]
        return target, lc, rc
