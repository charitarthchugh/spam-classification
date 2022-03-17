from transformers import BertTokenizer
from pathlib import Path

MAX_LEN = 512
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 8
EPOCHS = 10
ACCUMULATION = 2
BERT_PATH = Path("../input/bert_base_uncased")
MODEL_PATH = "model.bin"
TRAINING_FILE = Path('../data/train-data.csv')
TESTING_FILE = Path('../data/test-data.csv')
TOKENIZER = BertTokenizer.from_pretrained('bert-base-uncased')
 