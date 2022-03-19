import torch
from torch.utils.data import Dataset
from transformers.file_utils import PaddingStrategy
import config


class SpamDataset(Dataset):
    """Base Dataset class to inherit from, uses BERT tokenizer by default"""

    def __init__(self, texts, target):
        self.texts = texts
        self.target = target
        self.max_len = config.MAX_LEN
        self.tokenizer = config.TOKENIZER

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item) -> dict[str, torch.Tensor]:
        text = str(self.texts[item])
        text = " ".join(text.split())
        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding=PaddingStrategy("max_length"),
        )
        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]
        token_type_ids = inputs["token_type_ids"]
        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "target": torch.tensor(self.target[item], dtype=torch.float),
            "mask": torch.tensor(mask, dtype=torch.long),
        }


class DistilBertSpamDataset(SpamDataset):
    """
    Dataset for DistilBert based models (do not need token_type_ids)
    """

    def __getitem__(self, item) -> dict[str, torch.Tensor]:
        text = str(self.texts[item])
        text = " ".join(text.split())
        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding=PaddingStrategy("max_length"),
        )
        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]
        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "target": torch.tensor(self.target[item], dtype=torch.float),
            "mask": torch.tensor(mask, dtype=torch.long),
        }
