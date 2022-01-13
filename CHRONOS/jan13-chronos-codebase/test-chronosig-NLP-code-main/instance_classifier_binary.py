import pandas as pd
import numpy as np

from tqdm.auto import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import accuracy, f1, auroc
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger


from transformers import BertTokenizerFast as BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup
from transformers import RobertaTokenizerFast as RobertaTokenizer
from transformers import AutoTokenizer, AutoModel

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, multilabel_confusion_matrix

import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc

import argparse

from loguru import logger

# data class
class InstanceDataset(Dataset):
    def __init__(self,
        data: pd.DataFrame,
        tokenizer: BertTokenizer,
        max_token_len: int = 512, mode = "train"):

        self.tokenizer = tokenizer
        self.data = data
        self.max_token_len = max_token_len
        self.mode = mode
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        data_row = self.data.iloc[index]

        all_text = data_row["Clinical_Note_Text"]
        labels = data_row["rejection"]
        encoding = self.tokenizer.encode_plus(
          all_text,
          add_special_tokens=True,
          max_length=self.max_token_len,
          return_token_type_ids=False,
          padding="max_length",
          truncation=True,
          return_attention_mask=True,
          return_tensors='pt',
        )

        #TODO  - implement a balancing of the dataset - i.e. have around 50-50 split of labels

        return dict(
          all_text=all_text,
          input_ids=encoding["input_ids"].flatten(),
          attention_mask=encoding["attention_mask"].flatten(),
          labels=torch.tensor(labels)
        )

# data module class - wrapped around pytorch lightning data module
class InstanceDataModule(pl.LightningDataModule):
    def __init__(self, train_df, valid_df,test_df, tokenizer, batch_size=2, max_token_len=512):
        super().__init__()
        self.batch_size = batch_size
        self.train_df = train_df
        self.valid_df = valid_df
        self.test_df = test_df
        self.tokenizer = tokenizer
        self.max_token_len = max_token_len

        logger.warning(f"size of training dataset: {len(train_df)} ")
        logger.warning(f"size of validation dataset: {len(valid_df)} ")
        logger.warning(f"size of test dataset: {len(test_df)} ")




    def setup(self, stage=None):
        self.train_dataset = InstanceDataset(
            self.train_df,
            self.tokenizer,
            self.max_token_len
        )

        self.valid_dataset = InstanceDataset(
            self.valid_df,
            self.tokenizer,
            self.max_token_len
        )
        self.test_dataset = InstanceDataset(
            self.test_df,
            self.tokenizer,
            self.max_token_len
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True

        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_dataset,
            batch_size=self.batch_size

        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size

        )

# Bert model base

class InstanceBertModel(pl.LightningModule):
    def __init__(self,
                 bert_model,
                 num_labels,
                 bert_hidden_dim=768,
                 classifier_hidden_dim=768,
                 n_training_steps=None,
                 n_warmup_steps=None,
                 dropout=0.5,
                 weight_classes=False,
                 weights=torch.tensor([0.5, 1.5]),
                 reinit_n_layers=0,
                 pretrained_dir ="F:/OxfordTempProjects/PatientTriageNLP/nlp_development/pretrained_hf_models/" ):

        super().__init__()
        logger.warning(f"Building model based on following architecture. {bert_model}")
        self.num_labels = num_labels
        self.save_hyperparameters()
        self.bert = AutoModel.from_pretrained(f"{pretrained_dir}/{bert_model}/model", return_dict=True)
        # nn.Identity does nothing if the dropout is set to None
        self.classifier = nn.Sequential(nn.Linear(bert_hidden_dim, classifier_hidden_dim),
                                        nn.ReLU(),
                                        nn.Dropout(dropout) if dropout is not None else nn.Identity(),
                                        nn.Linear(classifier_hidden_dim, num_labels))
        #reinitialize n layers
        self.reinit_n_layers = reinit_n_layers
        if reinit_n_layers > 0:
            logger.warning(f"Re-initializing the last {reinit_n_layers} layers of encoder")
            self._do_reinit()
        #if we want to bias loss based on class sample sizes
        if weight_classes:
            self.criterion = nn.CrossEntropyLoss(weight=weights)
        else:
            self.criterion = nn.CrossEntropyLoss()

        self.n_training_steps = n_training_steps
        self.n_warmup_steps = n_warmup_steps

    def _do_reinit(self):
        # re-init pooler
        self.bert.pooler.dense.weight.data.normal_(mean=0.0, std=self.bert.config.initializer_range)
        self.bert.pooler.dense.bias.data.zero_()
        for param in self.bert.pooler.parameters():
            param.requires_grad = True

        # re-init last n layers
        for n in range(self.reinit_n_layers):
            self.bert.encoder.layer[-(n + 1)].apply(self._init_weight_and_bias)

    def _init_weight_and_bias(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.bert.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, input_ids, attention_mask, labels=None):
        output = self.bert(input_ids, attention_mask=attention_mask)
        # obtaining the last layer hidden states of the Transformer
        last_hidden_state = output.last_hidden_state  # shape: (batch_size, seq_length, bert_hidden_dim)

        #         or can use the output pooler : output = self.classifier(output.pooler_output)
        # As I said, the CLS token is in the beginning of the sequence. So, we grab its representation
        # by indexing the tensor containing the hidden representations
        CLS_token_state = last_hidden_state[:, 0, :]
        # passing this representation through our custom head
        logits = self.classifier(CLS_token_state)
        loss = 0
        if labels is not None:
            loss = self.criterion(logits, labels)
        return loss, logits

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, outputs = self(input_ids, attention_mask, labels)
        self.log("train_loss", loss, prog_bar=True, logger=True)
        return {"loss": loss, "predictions": outputs, "labels": labels}

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, outputs = self(input_ids, attention_mask, labels)
        self.log("val_loss", loss, prog_bar=True, logger=True)
        return {"loss": loss, "predictions": outputs, "labels": labels}

    def test_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, outputs = self(input_ids, attention_mask, labels)
        self.log("test_loss", loss, prog_bar=True, logger=True)
        return loss

    def training_epoch_end(self, outputs):
        softmax = nn.functional.softmax
        labels = []
        predictions = []
        for output in outputs:
            for out_labels in output["labels"].detach().cpu():
                labels.append(out_labels)
            for out_predictions in output["predictions"].detach().cpu():
                out_predictions = softmax(out_predictions)
                predictions.append(out_predictions)

        labels = torch.stack(labels).int()
        predictions = torch.stack(predictions)

        roc_auc = auroc(predictions, labels, num_classes=self.num_labels)
        f1_score = f1(predictions,labels,average = "weighted", num_classes = self.num_labels)

        self.logger.experiment.add_scalar(f"roc_auc/Train", roc_auc, self.current_epoch)
        self.logger.experiment.add_scalar(f"f1_weighted/Train", f1_score, self.current_epoch)


    def validation_epoch_end(self, outputs):
        softmax = nn.functional.softmax
        labels = []
        predictions = []
        for output in outputs:
            for out_labels in output["labels"].detach().cpu():
                labels.append(out_labels)
            for out_predictions in output["predictions"].detach().cpu():
                out_predictions = softmax(out_predictions)
                predictions.append(out_predictions)

        labels = torch.stack(labels).int()
        predictions = torch.stack(predictions)

        try:
            roc_auc = auroc(predictions, labels, num_classes=2)
            f1_score = f1(predictions, labels, average="weighted", num_classes=self.num_labels)
            self.logger.experiment.add_scalar(f"roc_auc/Valid", roc_auc, self.current_epoch)
            self.logger.experiment.add_scalar(f"f1_weighted/Valid", f1_score, self.current_epoch)
        except:
            logger.warning("validation epoch end: could not calculate roc/f1. Very likely due to small batch size and only one class in target")

    def test_epoch_end(self, outputs):
        softmax = nn.functional.softmax
        labels = []
        predictions = []
        for output in outputs:
            for out_labels in output["labels"].detach().cpu():
                labels.append(out_labels)
            for out_predictions in output["predictions"].detach().cpu():
                out_predictions = softmax(out_predictions)
                predictions.append(out_predictions)

        labels = torch.stack(labels).int()
        predictions = torch.stack(predictions)

        try:
            roc_auc = auroc(predictions, labels, num_classes=2)
            f1_score = f1(predictions, labels, average="weighted", num_classes=self.num_labels)
            self.logger.experiment.add_scalar(f"roc_auc/Test", roc_auc, self.current_epoch)
            self.logger.experiment.add_scalar(f"f1_weighted/Test", f1_score, self.current_epoch)
        except:
            logger.warning("validation epoch end: could not calculate roc/f1. Very likely due to small batch size and only one class in target")


    def configure_optimizers(self):

        optimizer = AdamW(self.parameters(), lr=2e-5)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.n_warmup_steps,
            num_training_steps=self.n_training_steps
        )
        logger.warning(f"Optimizer set up with the following parameters: {optimizer}")
        return dict(
            optimizer=optimizer,
            lr_scheduler=dict(
                scheduler=scheduler,
                interval='step'
            )
        )


