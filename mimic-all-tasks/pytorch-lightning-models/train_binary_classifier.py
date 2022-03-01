import os

import pandas as pd
import numpy as np
from tqdm.auto import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from loguru import logger

from transformers import BertTokenizerFast as BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup
from transformers import RobertaTokenizerFast as RobertaTokenizer
from transformers import AutoTokenizer, AutoModel
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, multilabel_confusion_matrix

from bert_classifier import MimicBertModel, MimicDataset, MimicDataModule
import argparse
from datetime import datetime


'''
Script to run training with a argument specified BERT model as the pre-trained encoder for instance classification.


Example cmd usage:

python train_binary_classifier.py --transformer_type bert --encoder_model emilyalsentzer/Bio_ClinicalBERT --batch_size 4 --gpus 0 --max_epochs 10 --dataset mortality --bert_model emilyalsentzer/Bio_ClinicalBERT --fast_dev_run True
'''

# classes are imbalanced - lets calculate class weights for loss

def get_class_weights(train_df, label_col):
    classes = list(train_df[label_col].unique())
    class_dict = {}
    nSamples = []
    for c in classes:
        class_dict[c] = len(train_df[train_df[label_col] == c])
        nSamples.append(class_dict[c])

    normedWeights = [1 - (x / sum(nSamples)) for x in nSamples]
    return torch.FloatTensor(normedWeights)

def read_csv(data_dir,filename):
    return pd.read_csv(f"{data_dir}{filename}", index_col=None)

def main():
    parser = argparse.ArgumentParser()

    #TODO - add an argument to specify whether using balanced data then update directories based on that

    # Required parameters
    parser.add_argument("--data_dir",
                        default = "../clinical-outcomes-data/mimic3-clinical-outcomes/mp/",
                        type=str,
                        help = "The data path to the directory containing the notes and referral data files")

    parser.add_argument("--training_file",
                        default = "train.csv",
                        type=str,
                        help = "The data path to the directory containing the notes and referral data files")
    parser.add_argument("--validation_file",
                        default = "valid.csv",
                        type=str,
                        help = "The default name of the training file")
    parser.add_argument("--test_file",
                        default = "test.csv",
                        type=str,
                        help = "The default name of hte test file")

    parser.add_argument("--pretrained_models_dir",
                        default="F:/OxfordTempProjects/PatientTriageNLP/nlp_development/pretrained_hf_models/",
                        type=str,
                        help="The data path to the directory containing local pretrained models from huggingface")
    parser.add_argument("--bert_model",
                        default="emilyalsentzer/Bio_ClinicalBERT",
                        type=str,
                        help="bert encoder architecture - e.g. bert-base-uncased")

    parser.add_argument("--text_col",
                        default = "Clinical_Note_Text",
                        type=str,
                        help = "The data path to the directory containing the notes and referral data files")

    parser.add_argument("--save_dir",
                        default = "F:/OxfordTempProjects/PatientTriageNLP/experiments/binary_instance_classification/pytorch-lightning-models/",
                        type=str,
                        help = "The data path to the directory containing the notes and referral data files"
                        )
    parser.add_argument("--num_classes",
                        default = 2,
                        type=int,
                        help = "number of classes for classification problem"
                        )

    parser.add_argument("--reinit_n_layers",
                        default = 0,
                        type=int,
                        help = "number of pretrained final bert encoder layers to reinitialize for stabilisation"
                        )
    parser.add_argument("--max_tokens",
                        default = 512,
                        type=int,
                        help = "Max tokens to be used in modelling"
                        )
    parser.add_argument("--max_epochs",
                        default = 30,
                        type=int,
                        help = "Number of epochs to train"
                        )
    parser.add_argument("--batch_size",
                        default = 4,
                        type=int,
                        help = "batch size for training"
                        )
    parser.add_argument("--accumulate_grad_batches",
                        default = 4,
                        type=int,
                        help = "number of batches to accumlate before optimization step"
                        )
    parser.add_argument("--balance_data",
                        action = 'store_true',
                        help="Whether not to balance dataset based on least sampled class")

    parser.add_argument("--gpus", type=int, default=1, help="Which gpu device to use e.g. 0 for cuda:0, or for more gpus use comma separated e.g. 0,1,2")

    parser.add_argument(
            "--encoder_model",
            default= 'emilyalsentzer/Bio_ClinicalBERT',# 'allenai/biomed_roberta_base',#'simonlevine/biomed_roberta_base-4096-speedfix', # 'bert-base-uncased',
            type=str,
            help="Encoder model to be used.",
        )

    parser.add_argument(
        "--transformer_type",
        default='bert', #'longformer', roberta-long
        type=str,
        help="Encoder model /tokenizer to be used (has consequences for tokenization and encoding; default = longformer).",
    )   
    
    parser.add_argument(
        "--max_tokens_longformer",
        default=4096,
        type=int,
        help="Max tokens to be considered per instance..",
    )

    parser.add_argument(
        "--encoder_learning_rate",
        default=1e-05,
        type=float,
        help="Encoder specific learning rate.",
    )
    parser.add_argument(
        "--classifier_learning_rate",
        default=3e-05,
        type=float,
        help="Classification head learning rate.",
    )
    parser.add_argument(
        "--nr_frozen_epochs",
        default=0,
        type=int,
        help="Number of epochs we want to keep the encoder model frozen.",
    )

    parser.add_argument(
        "--dataset",
        default="mortality", #or: icd9_triage
        type=str,
        help="name of dataset",
    )

    parser.add_argument(
        "--label_col",
        default="hospital_expire_flag", #or: label/rejection/readmission etc
        type=str,
        help="string value of column name with the int class labels",
    )

    parser.add_argument(
        "--loader_workers",
        default=24,
        type=int,
        help="How many subprocesses to use for data loading. 0 means that \
            the data will be loaded in the main process.",
    )

    # Early Stopping
    parser.add_argument(
        "--monitor", default="monitor_balanced_accuracy", type=str, help="Quantity to monitor."
    )

    parser.add_argument(
        "--metric_mode",
        default="max",
        type=str,
        help="If we want to min/max the monitored quantity.",
        choices=["auto", "min", "max"],
    )
    parser.add_argument(
        "--patience",
        default=5,
        type=int,
        help=(
            "Number of epochs with no improvement "
            "after which training will be stopped."
        ),
    )
    parser.add_argument(
        '--fast_dev_run',
        default=False,
        type=bool,
        help='Run for a trivial single batch and single epoch.'
    )

    parser.add_argument(
        "--optimizer",
        default="adamw",
        type=str,
        help="Optimization algorithm to use e.g. adamw, adafactor",
    )
                        

    # TODO - add an argument to specify whether using balanced data then update directories based on that
    args = parser.parse_args()

    print(f"arguments provided are: {args}")
    # set up parameters
    data_dir = args.data_dir
    save_dir = args.save_dir
    pretrained_dir = args.pretrained_models_dir
    pretrained_model_name = args.bert_model
    max_tokens = args.max_tokens
    n_epochs = args.max_epochs
    batch_size = args.batch_size
    n_labels = args.num_classes
    reinit_n_layers = args.reinit_n_layers
    accumulate_grad_batches = args.accumulate_grad_batches

    time_now = datetime.now().strftime("%d-%m-%Y--%H-%M-%S")
    # set up the ckpt and logging dirs



    # update ckpt and logs dir based on the dataset
    
    ckpt_dir = f"./ckpts/{args.dataset}/{args.encoder_model}/version_{time_now}"
    log_dir = f"./logs/{args.dataset}/"

    # update ckpt and logs dir based on whether plm (encoder) was frozen during training

    if args.nr_frozen_epochs > 0:
        logger.warning(f"Freezing the encoder/plm for {args.nr_frozen_epochs} epochs")
        ckpt_dir = f"../ckpts/{args.dataset}/frozen_plm/{args.encoder_model}/version_{time_now}"
        log_dir = f"../logs/{args.dataset}/frozen_plm/"

    # load tokenizer
    print(f"loading tokenizer : {pretrained_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(f"{pretrained_model_name}")

    # read in training and validation data
    train_df = read_csv(data_dir, args.training_file)
    val_df = read_csv(data_dir, args.validation_file)
    test_df = read_csv(data_dir, args.test_file)

    logger.warning(f"train_df shape: {train_df.shape} and train_df cols:{train_df.columns}")

    # push data through pipeline
    # instantiate datamodule
    data_module = MimicDataModule(
        train_df,
        val_df,
        test_df,
        tokenizer,
        batch_size=batch_size,
        max_token_len=max_tokens,
        label_col = args.label_col,
        num_workers=args.loader_workers,
    )

    logger.warning(f"datamodule is: {data_module}")
    # define class_labels - i.e. the raw/string version of class labels such as "alive/deceased" for mortality prediction. 
    # These class_labels should be in a list where their index reflects the numerical class number
    class_labels = ["alive","deceased"]

    steps_per_epoch = len(train_df) // batch_size
    total_training_steps = steps_per_epoch * n_epochs
    warmup_steps = total_training_steps // 5
    warmup_steps, total_training_steps

    # get some class specific loss weights - only needed if doing some form of weighted cross entropy with ubalanced classes
    ce_class_weights = get_class_weights(data_module.train_df, args.label_col)

    #set up model
    model = MimicBertModel(bert_model=pretrained_model_name,
                                 num_labels=n_labels,
                                 n_warmup_steps=warmup_steps,
                                 n_training_steps=total_training_steps,
                                 nr_frozen_epochs = args.nr_frozen_epochs,
                                 ce_class_weights=ce_class_weights,
                                 weight_classes=True,
                                 reinit_n_layers=reinit_n_layers,
                                 class_labels = class_labels,
                                 encoder_learning_rate = args.encoder_learning_rate,
                                 classifier_learning_rate = args.classifier_learning_rate,
                                 optimizer= args.optimizer                            
                                 )

    #setup checkpoint and logger
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{ckpt_dir}",
        filename="best-checkpoint",
        save_top_k=1,
        verbose=True,
        monitor=args.monitor,
        mode=args.metric_mode,
        save_last = True
    )

    tb_logger = TensorBoardLogger(
        save_dir=f"{log_dir}",
        version="version_" + time_now,
        name=f'{args.encoder_model}',
    )

    # early stopping based on validation metric of choice
    early_stopping_callback = EarlyStopping(monitor=args.monitor, mode = args.metric_mode, patience=args.patience)

    # ------------------------
    # 5 INIT TRAINER
    # ------------------------
    trainer = Trainer(
        logger=tb_logger,
        gpus=[args.gpus],
        log_gpu_memory="all",
        fast_dev_run=args.fast_dev_run,
        accumulate_grad_batches=args.accumulate_grad_batches,
        checkpoint_callback = checkpoint_callback,
        callbacks = [early_stopping_callback],
        max_epochs=args.max_epochs,
        default_root_dir=f'./'        
    )

    print(f"trainer is {trainer}")
    # ------------------------
    # 6 START TRAINING
    # ------------------------

    # datamodule = MedNLIDataModule
    trainer.fit(model, data_module)

    # test
    # trainer.test(ckpt_path="best")

# run script
if __name__ == "__main__":
    main()