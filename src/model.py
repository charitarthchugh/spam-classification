from torch import nn
from transformers.models.bert import BertModel, modeling_bert
from transformers.models.distilbert import DistilBertModel, modeling_distilbert


class TransformerModel(nn.Module):
    """Base features for a simple transformers based model"""

    def __init__(self):
        super().__init__()
        self.dropout = nn.Dropout(p=0.3)
        self.out = nn.Linear(768, 1)


class BERTBaseUncased(TransformerModel):
    """Train a bert-base model"""

    def __init__(self):
        super().__init__()
        self.bert: modeling_bert.BertModel = BertModel.from_pretrained(
            "bert-base-uncased"
        )

    def forward(self, ids, mask, token_type_ids):
        """Interate forward in the training loop

        Args:
            ids: from tokenizer
            mask: from tokenizer
            token_type_ids: from tokenizer

        Returns:
            None
        """
        _, out = self.bert(
            ids, attention_mask=mask, token_type_ids=token_type_ids, return_dict=False
        )
        bo = self.dropout(out)
        out = self.out(bo)
        return out


class DistilBertBaseUncased(TransformerModel):
    """Train a distilbert-base model"""

    def __init__(self):
        super().__init__()
        self.distilbert: modeling_distilbert.DistilBertModel = (
            DistilBertModel.from_pretrained("distilbert-base-uncased")
        )

    def forward(self, ids, mask):
        """Interate forward in the training loop

        Args:
            ids: from tokenizer
            mask: from tokenizer

        Returns:
            None
        """
        _, out = self.distilbert(ids, attention_mask=mask, return_dict=False)
        return self.out(self.dropout(out))
