import torch
from sklearn.model_selection import train_test_split


class LemmatizationDataset(torch.utils.data.Dataset):

    def __init__(self, char_dataset):
        self.input_token2idx = self.create_token2idx("input", char_dataset)
        self.target_token2idx = self.create_token2idx("target", char_dataset)

        self.dataset = self.tokenize_dataset(char_dataset)

    @staticmethod
    def create_token2idx(dest, char_dataset):
        chars = set.union(*map(lambda x: set(x[dest]), char_dataset))
        if dest == "input":
            token2idx = {char: idx for idx, char in enumerate(chars, start=1)}
        elif dest == "target":
            token2idx = {char: idx for idx, char in enumerate(chars, start=3)}
            token2idx["<BOS>"] = 1
            token2idx["<EOS>"] = 2
        else:
            raise ValueError(f"Invalid destination {dest}")
        token2idx["<PAD>"] = 0
        return token2idx

    def tokenize_dataset(self, char_dataset):
        result = []
        for data_point in char_dataset:

            cur_inp = []
            for char in data_point["input"]:
                cur_inp.append(self.input_token2idx[char])

            cur_targ = []
            for char in data_point["target"]:
                cur_targ.append(self.target_token2idx[char])
            cur_targ = [self.target_token2idx["<BOS>"]] + cur_targ + [self.target_token2idx["<EOS>"]]

            result.append(
                {
                    "input": cur_inp,
                    "target": cur_targ,
                }
            )
        return result

    def __getitem__(self, idx):
        return self.dataset[idx]

    def __len__(self):
        return len(self.dataset)


def collate_fn(batch):
    inputs = []
    targets = []

    for data_point in batch:
        inputs.append(torch.LongTensor(data_point["input"]))
        targets.append(torch.LongTensor(data_point["target"]))

    inp_tensor = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True)
    target_tensor = torch.nn.utils.rnn.pad_sequence(targets, batch_first=True)

    return inp_tensor, target_tensor


def get_data_loaders(dataset: LemmatizationDataset, batch_size: int, test_size: float):
    train, valid = train_test_split(dataset, test_size=test_size)
    train_loader = torch.utils.data.DataLoader(
        train,
        shuffle=True,
        batch_size=batch_size,
        collate_fn=collate_fn
    )
    valid_loader = torch.utils.data.DataLoader(
        valid,
        shuffle=True,
        batch_size=batch_size,
        collate_fn=collate_fn
    )
    return {
        "train": train_loader,
        "valid": valid_loader,
    }
