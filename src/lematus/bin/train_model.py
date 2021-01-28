import argparse
import pickle
import yaml

from lematus.data_utils.dataset import LemmatizationDataset, get_data_loaders
from lematus.model.model import ConditionalSeq2SeqModel
from lematus.model.training.trainer import Trainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t", "--train-config-path", type=str, default="config/train_config.yaml", help="Path to train config",
    )
    parser.add_argument(
        "-m", "--model-config-path", type=str, default="config/model_config.yaml", help="Path to model config",
    )
    parser.add_argument(
        "-d", "--device", type=str, default='cpu', help="Device to put model",
    )
    parser.add_argument(
        "-b", "--batch-size", type=int, default=64, help="Batch size to train model",
    )
    parser.add_argument(
        "--dataset-path", type=str, default="data/dataset.pkl", help="Path to training data",
    )
    parser.add_argument(
        "--test-size", type=float, default=0.25, help="Test data fraction",
    )
    args = parser.parse_args()
    with open(args.train_config_path, 'r') as f:
        train_config = yaml.load(f)
    with open(args.model_config_path, 'r') as f:
        model_config = yaml.load(f)
    with open(args.dataset_path, 'rb') as f:
        dataset = pickle.load(f)
    lemmatization_dataset = LemmatizationDataset(dataset)
    dataloaders = get_data_loaders(lemmatization_dataset, args.batch_size, args.test_size)

    model = ConditionalSeq2SeqModel(
        model_config['encoder_params'],
        model_config['decoder_params'],
        lemmatization_dataset.input_token2idx,
        lemmatization_dataset.target_token2idx,
    )

    trainer = Trainer(model, args.device)
    trainer.train_model(dataloaders, **train_config)


if __name__ == "__main__":
    main()
