from torch.utils.data import DataLoader, Dataset
import torch
from torch.utils.data.dataset import T_co


class DeviceDataLoader(DataLoader):
    def __init__(self, dataset: Dataset[T_co]):
        super().__init__(dataset)
        if torch.cuda.is_available():
            print(torch.device)
