import argparse
import pickle

from lematus.data_utils.preprocessing import Preprocessor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-w", "--window-size", type=int, help="Context window size",
    )
    parser.add_argument(
        "-i", "--input-file", type=str, default="data/ru_syntagrus-ud-train.conllu", help="Path to conllu dataset",
    )
    parser.add_argument(
        "-o", "--output-file", type=str, default="data/dataset.pkl", help="Path to output dataset",
    )
    parser.add_argument(
        '-l', "--language", type=str, default='rus', help="Language of conllu dataset",
    )
    args = parser.parse_args()
    preprocessor = Preprocessor(args.language)

    dataset = preprocessor.get_dataset_from_file(args.input_file, args.window_size)

    with open(args.output_file, 'wb') as f:
        pickle.dump(dataset, f)


if __name__ == "__main__":
    main()
