import torch
from torch import nn


class SpamModel(nn.Module):

    def __init__(self, vocab_size, embed_dim, num_class):
        super(SpamModel, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
        self.fc = nn.Linear(embed_dim, num_class)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        return self.fc(embedded)

# class SpamModel(nn.Module):
#     def __init__(self, vocab_size, embed_dim):
#         super(SpamModel, self).__init__()
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
#         self.fc = nn.Linear(embed_dim, 2)
#         self.init_weights()
#
#     def init_weights(self):
#         initrange = 0.5
#         self.embedding.weight.data.uniform_(-initrange, initrange)
#         self.fc.weight.data.uniform_(-initrange,initrange)
#         self.fc.bias.data.zero_()
#
#     def loss(self, outputs, targets):
#         return nn.BCEWithLogitsLoss()(outputs, targets.view(-1, 1))
#
#     def monitor_metrics(self, outputs, targets):
#         outputs = torch.sigmoid(outputs).cpu().detach().numpy() >= 0.5
#         targets = targets.cpu().detarch.numpy()
#         return {
#             "accuracy": metrics.accuracy_score(targets, outputs),
#         }
#
#     def fetch_optimizer(self):
#         opt = AdamW(self.parameters(), lr=1e-4)
#         return opt
#
#     def fetch_scheduler(self):
#         sch = get_linear_schedule_with_warmup(
#             self.optimizer, num_warmup_steps=0, num_training_steps=self.train_steps
#         )
#         return sch
#
#     def forward(self, ids, mask, token_type_ids, targets=None):
#         _, x = self.bert(
#             ids,
#             attention_mask=mask,
#             token_type_ids=token_type_ids,
#         )
#         x = self.bert_dropout(0)
#         x = self.out(x)
#         if targets is not None:
#             loss = self.loss(x, targets)
#             met = self.metrics(x, targets)
#             return x, loss, met
#         return x, -1, {}
