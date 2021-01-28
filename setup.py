from setuptools import setup, find_packages


def main():
    with open("requirements.txt") as f:
        requirements = f.read()

    console_scripts = [
        "create_dataset_from_conllu_format = lematus.bin.create_dataset_from_conllu_format:main",
        "train_model = lematus.bin.train_model:main"
    ]

    setup(
        name="lematus",
        version="0.1",
        author="Philipp Fisin",
        package_dir={"": "src"},
        packages=find_packages("src"),
        description="Pure pytorch implementation of conditional seq2seq model for lemmatization task",
        install_requires=requirements,
        entry_points={
            "console_scripts": console_scripts
        }
    )


if __name__ == "__main__":
    main()
