import io
from pathlib import Path


from torch.utils.data import IterableDataset
import pandas as pd
from torchtext.utils import unicode_csv_reader

PATHS = {"train": r"data/train-data.csv", "valid": r"data/valid-data.csv"}


class RawTextIterableDataset(IterableDataset):
    """Defines an abstraction for raw text iterable datasets."""

    def __init__(self, full_num_lines, iterator):
        """Initiate the dataset abstraction."""
        super(RawTextIterableDataset, self).__init__()
        self.full_num_lines = full_num_lines
        self._iterator = iterator
        self.num_lines = full_num_lines
        self.current_pos = None

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_pos == self.num_lines - 1:
            raise StopIteration
        item = next(self._iterator)
        if self.current_pos is None:
            self.current_pos = 0
        else:
            self.current_pos += 1
        return item

    def __len__(self):
        return self.num_lines

    def pos(self):
        """
        Returns current position of the iterator. This returns None
        if the iterator hasn't been used yet.
        """
        return self.current_pos


def spam_dataset(split: str):
    def _create_data_from_csv(data_path):
        with io.open(data_path, encoding="utf8") as f:
            reader = unicode_csv_reader(f)
            for row in reader:
                yield int(row[0]), " ".join(row[1:])

    NUM_LINES = len(pd.read_csv(PATHS[split]).axes[0])
    return RawTextIterableDataset(NUM_LINES, _create_data_from_csv(PATHS[split]))