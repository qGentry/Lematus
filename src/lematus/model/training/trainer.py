import torch
import os

from lematus.model.model import ConditionalSeq2SeqModel
from lematus.nn_utils.losses import calculate_seq_loss
from lematus.nn_utils.metrics import calc_accuracy


class Trainer:

    def __init__(self, model: ConditionalSeq2SeqModel, device: str = 'cpu'):
        self.device = device
        self.model = model
        self.model.to(device)

    def train_model(self,
                    dataloaders: dict,
                    model_dir: str,
                    optimizer_config: dict,
                    num_epochs: int = 3,
                    train_log_steps: int = 100,
                    ):
        os.makedirs(model_dir)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=optimizer_config["learning_rate"])
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, optimizer_config['step_lr_steps'], gamma=optimizer_config['gamma'],
        )
        for k in range(num_epochs):
            print(f"Starting epoch {k}")
            self.model.train()
            losses = []
            accs = []
            for i, batch in enumerate(dataloaders['train']):
                optimizer.zero_grad()
                inp, target = batch
                target = target.to(self.device)
                preds, lens, _ = self.model(inp, true_labels=target, train=True)

                loss = calculate_seq_loss(preds, target)
                losses.append(loss.item())

                acc = calc_accuracy(preds, target)
                accs.append(acc)

                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), optimizer_config['clip_norm'])
                optimizer.step()

                if i % train_log_steps == 0:
                    print(f"train loss step {i} = {sum(losses) / len(losses)}")
                    print(f"train accs step {i} = {sum(accs) / len(accs)}")
                    print()
                    losses = []
                    accs = []
            valid_loss, valid_acc = self.validate_model(dataloaders['valid'])
            print(f"valid loss epoch {k} = {valid_loss}")
            print(f"valid accs epoch {k} = {valid_acc}")
            self.model.save_model()
            lr_scheduler.step()

    def validate_model(self, validation_dataset):
        self.model.eval()
        losses = []
        accs = []
        for i, batch in enumerate(validation_dataset):
            inp, target = batch
            target = target.to(self.device)
            preds, lens, _ = self.model(inp, true_labels=target, train=False)

            loss = calculate_seq_loss(preds, target)
            losses.append(loss.item())

            acc = calc_accuracy(preds, target)
            accs.append(acc)
        return sum(losses) / len(losses), sum(accs) / len(accs)
