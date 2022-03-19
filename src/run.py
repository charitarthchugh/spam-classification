import numpy as np
import pandas as pd
import torch
import transformers
from sklearn import metrics
from sklearn.model_selection import train_test_split
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

import config
from .dataset import SpamDataset
from .model import BERTBaseUncased


def loss_fn(outputs, targets):
    return nn.BCEWithLogitsLoss()(outputs.reshape(-1, outputs.shape[0])[0], targets)


def train_fn(
    data_loader: DataLoader,
    model: nn.Module,
    optimizer,
    device: torch.device,
    scheduler=transformers.get_scheduler,
    fp16: bool = False,
):
    model.train()
    for bi, d in tqdm(enumerate(data_loader), total=len(data_loader)):
        ids = d["ids"].to(device, dtype=torch.long)
        token_type_ids = d["token_type_ids"].to(device, dtype=torch.long)
        mask = d["mask"].to(device, dtype=torch.long)
        targets = d["target"].to(device, dtype=torch.float16)

        optimizer.zero_grad()
        if fp16:
            with torch.cuda.amp.autocast():
                outputs = model(ids=ids, mask=mask, token_type_ids=token_type_ids)
                loss = loss_fn(outputs, targets)
            config.SCALER.scale(loss).backward()
            config.SCALER.step(optimizer)
            config.SCALER.update()
        else:
            outputs = model(ids=ids, mask=mask, token_type_ids=token_type_ids)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()
        scheduler.step()


def eval_fn(
    data_loader: torch.utils.data.DataLoader,
    model: nn.Module,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> tuple[list[float], list[float]]:
    model.eval()
    fin_targets = []
    fin_outputs = []
    with torch.no_grad():
        for b, d in tqdm(enumerate(data_loader), total=len(data_loader)):
            ids = d["ids"]
            token_type_ids = d["token_type_ids"]
            mask = d["mask"]
            targets = d["target"]

            ids = ids.to(device, dtype=torch.long)
            token_type_ids = token_type_ids.to(device, dtype=torch.long)
            mask = mask.to(device, dtype=torch.long)
            targets = targets.to(device, dtype=torch.float)

            # ids, token_type_ids, mask = lazy(ids, token_type_ids, mask, batch=0)
            # targets = lazy(targets)

            outputs = model(ids=ids, mask=mask, token_type_ids=token_type_ids)
            fin_targets.extend(targets.cpu().detach().numpy().tolist())
            fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
    return fin_outputs, fin_targets


def run():
    df = pd.read_csv(config.TRAINING_FILE)
    df_train, df_valid = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df.targets.values,
    )
    df_train.reset_index(drop=True, inplace=True)
    df_valid.reset_index(drop=True, inplace=True)

    train_ds = SpamDataset(texts=df_train.texts.values, target=df_train.targets.values)
    train_dl = DataLoader(
        dataset=train_ds,
        batch_size=config.TRAIN_BATCH_SIZE,
        num_workers=4,
        pin_memory=True,
    )
    valid_ds = SpamDataset(
        texts=df_valid.texts.values,
        target=df_valid.targets.values,
    )
    valid_dl = DataLoader(
        dataset=valid_ds,
        batch_size=config.VALID_BATCH_SIZE,
        num_workers=1,
        pin_memory=True,
    )
    device = torch.device("cuda")
    model = BERTBaseUncased().to(device)
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.001,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    num_train_steps = int(len(df_train) / config.TRAIN_BATCH_SIZE * config.EPOCHS)
    optimizer = AdamW(optimizer_parameters, lr=3e-5)
    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_train_steps
    )
    best_accuracy = 0
    for epochs in range(config.EPOCHS):
        float16 = True
        train_fn(
            data_loader=train_dl,
            model=model,
            optimizer=optimizer,
            device=device,
            scheduler=scheduler,
            fp16=True,
        )
        outputs, targets = eval_fn(valid_dl, model, device)
        outputs = np.array(outputs) >= 0.5
        accuracy = metrics.accuracy_score(targets, outputs)
        print(f"Accuracy Score = {accuracy}")
        if accuracy > best_accuracy:
            torch.save(model.state_dict(), f"{epochs}-{config.MODEL_PATH}")
            best_accuracy = accuracy
