from pathlib import Path

import pandas as pd
import torchtext.data.utils

from src.model import SpamModel
from src.dataset import spam_dataset
from torchtext.vocab import build_vocab_from_iterator

#
# def ham_to_binary():
#     if not Path.exists(Path("../data/train-data.csv")):
#         a = pd.read_csv(
#             "../data/train-data.tsv",
#             sep="\t",
#         )
#         a.columns = ["targets", "msg"]
#         a["targets"].replace(to_replace="ham", value="0", inplace=True)
#         a["targets"].replace(to_replace="spam", value="1", inplace=True)
#         a.to_csv("data/train-data.csv", index=False)
#     if not Path.exists(Path("../data/valid-data.csv")):
#         a = pd.read_csv(
#             "../data/valid-data.tsv",
#             sep="\t",
#         )
#         a.columns = ["targets", "msg"]
#         a["targets"].replace(to_replace="ham", value="0", inplace=True)
#         a["targets"].replace(to_replace="spam", value="1", inplace=True)
#         a.to_csv("data/valid-data.csv", index=False)
#
# def main_old():
#     # ham_to_binary()
#     train_df = pd.read_csv("../data/train-data.tsv", sep="\t")
#     valid_df = pd.read_csv("../data/valid-data.tsv", sep="\t")
#     train_dataset = SpamDataset(
#         texts=train_df["msg"].values, targets=train_df["targets"].values
#     )
#     valid_dataset = SpamDataset(
#         texts=valid_df["msg"].values, targets=valid_df["targets"].values
#     )
#     n_train_steps = int(len(train_df) / 32 * 10)
#     model = SpamModel(num_classes=1, num_train_steps=n_train_steps)
#     early_stop = tez.callbacks.EarlyStopping(
#         monitor="valid_loss", patience=3, model_path="model.bin"
#     )
#     model.fit(
#         train_dataset,
#         valid_dataset=valid_dataset,
#         epochs=10,
#         train_bs=32,
#         device="cuda",
#         callbacks=[early_stop],
#     )
#     model.load("model.bin", device="cuda")
#     preds = model.predict(dataset=valid_dataset)
#
tokenizer = torchtext.data.utils.get_tokenizer("spacy")


def yield_tokens(data_iter):
    for _, text in data_iter:
        yield tokenizer(text)


def ham_to_binary():
    if not Path.exists(Path("data/train-data.csv")):
        a = pd.read_csv(
            "data/train-data.tsv",
            sep="\t",
        )
        a.columns = ["targets", "msg"]
        a["targets"].replace(to_replace="ham", value="0", inplace=True)
        a["targets"].replace(to_replace="spam", value="1", inplace=True)
        a.to_csv("data/train-data.csv", index=False)
    if not Path.exists(Path("data/valid-data.csv")):
        a = pd.read_csv(
            "data/valid-data.tsv",
            sep="\t",
        )
        a.columns = ["targets", "msg"]
        a["targets"].replace(to_replace="ham", value=str(0), inplace=True)
        a["targets"].replace(to_replace="spam", value=str(1), inplace=True)
        a.to_csv("data/valid-data.csv", index=False)


def main():
    train_iter = spam_dataset(split="train")
    vocab = vocab = build_vocab_from_iterator(
        yield_tokens(train_iter),
    )
    vocab.set_default_index(vocab["<unk>"])


if __name__ == "__main__":
    ham_to_binary()
