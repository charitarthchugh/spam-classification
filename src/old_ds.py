from transformers import BertTokenizer


class SpamDataset:
    def __init__(self, texts, targets, max_len=64):
        self.texts = texts
        self.targets = targets
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def _getitem__(self, index):
        txt = str(self.texts[index])
        inputs = self.tokenizer.encode_plus(
            txt,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
        )