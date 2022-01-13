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
from sklearn.metrics import classification_report, precision_recall_curve, roc_curve, auc, roc_auc_score
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns

from instance_classifier_binary import InstanceBertModel, InstanceDataset, InstanceDataModule
import argparse
from datetime import datetime


'''
Script to evaluate a trained BERT model on a given dataset with appropriate metrics saved/plotted as output


Example usage:

'''



def read_csv(data_dir,filename):
    return pd.read_csv(f"{data_dir}{filename}", index_col=None)


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--data_dir",
                        default="F:/OxfordTempProjects/PatientTriageNLP/processed_data/instance_classification/balanced/concat/",
                        type=str,
                        help="The data path to the directory containing the notes and referral data files")

    parser.add_argument("--training_file",
                        default="train.csv",
                        type=str,
                        help="The data path to the directory containing the notes and referral data files")
    parser.add_argument("--validation_file",
                        default="valid.csv",
                        type=str,
                        help="The default name of the training file")
    parser.add_argument("--test_file",
                        default="test.csv",
                        type=str,
                        help="The default name of hte test file")

    parser.add_argument("--data_file",
                        default="F:/OxfordTempProjects/PatientTriageNLP/processed_data/processed_pre_referral_instances_concat.csv",
                        type=str,
                        help="The data path to the directory containing the notes and referral data files")
    parser.add_argument("--pretrained_models_dir",
                        default="F:/OxfordTempProjects/PatientTriageNLP/nlp_development/pretrained_hf_models/",
                        type=str,
                        help="The data path to the directory containing local pretrained models from huggingface")
    parser.add_argument("--bert_model",
                        default="bert-base-uncased",
                        type=str,
                        help="bert encoder architecture - e.g. bert-base-uncased")

    parser.add_argument("--text_col",
                        default="Clinical_Note_Text",
                        type=str,
                        help="The data path to the directory containing the notes and referral data files")

    parser.add_argument("--save_dir",
                        default="F:/OxfordTempProjects/PatientTriageNLP/processed_data/instance_classification/",
                        type=str,
                        help="The data path to the directory containing the notes and referral data files"
                        )
    parser.add_argument("--ckpt_dir",
                        default="F:/OxfordTempProjects/PatientTriageNLP/experiments/binary_instance_classification/pytorch-lightning-models/checkpoints/balanced/emilyalsentzer/Bio_ClinicalBERT/",
                        type=str,
                        help="The data path to the directory containing the trained model to be tested"
                        )
    parser.add_argument("--num_classes",
                        default=2,
                        type=int,
                        help="number of classes for classification problem"
                        )
    parser.add_argument("--max_tokens",
                        default=512,
                        type=int,
                        help="Max tokens to be used in modelling"
                        )
    parser.add_argument("--num_epochs",
                        default=10,
                        type=int,
                        help="Number of epochs to train"
                        )
    parser.add_argument("--batch_size",
                        default=2,
                        type=int,
                        help="batch size for training"
                        )

    args = parser.parse_args()

    print(f"arguments provided are: {args}")
    # set up parameters
    data_dir = args.data_dir
    pretrained_dir = args.pretrained_models_dir
    pretrained_model_name = args.bert_model
    # checkpoint dir
    ckpt_dir = args.ckpt_dir

    max_tokens = args.max_tokens
    n_epochs = args.num_epochs
    batch_size = args.batch_size
    n_labels = args.num_classes
    # load tokenizer
    print(f"loading tokenizer : {pretrained_dir}{pretrained_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(f"{pretrained_dir}/{pretrained_model_name}/tokenizer")

    # read in training and validation data
    train_df = read_csv(data_dir, args.training_file)
    val_df = read_csv(data_dir, args.validation_file)
    test_df = read_csv(data_dir, args.test_file)

    # combine val and test dfs into one large test set
    all_test = pd.concat([val_df, test_df])

    data_module = InstanceDataModule(
        train_df,
        val_df,
        test_df,
        tokenizer,
        batch_size=batch_size,
        max_token_len=max_tokens
    )

    # load trained model
    trained_model = InstanceBertModel.load_from_checkpoint(
      f"{ckpt_dir}/best-checkpoint.ckpt"
    )

    # set to eval mode to disable certain inter layer functions not required
    trained_model.eval()
    # freeze all layers - i.e. sets requires_grad=false
    trained_model.freeze()

    # if a cuda ready device i.e. gpu is available, transfer model to that, otherwise use cpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trained_model = trained_model.to(device)

    # load the datasets into data modules
    train_dataset = InstanceDataset(
      train_df,
      tokenizer,
      max_token_len=max_tokens
    )
    val_dataset = InstanceDataset(
      val_df,
      tokenizer,
      max_token_len=max_tokens
    )
    test_dataset = InstanceDataset(
      test_df,
      tokenizer,
      max_token_len=max_tokens
    )
    all_test_dataset = InstanceDataset(
        all_test,
        tokenizer,
        max_token_len=max_tokens
    )

    # crude - but put datasets of interest in array to iterate over
    datasets = [all_test_dataset]

    for name, ds in zip(["all_test_dataset"],datasets):

        print(f"Calculating evaluation metrics for :{name}")
        predictions = []
        labels = []

        # IMPORTANT - may be overkill at this point, but this ensures no gradients are calculated for any parameter - very important for memory usage during inference
        with torch.no_grad():
            for item in tqdm(ds):
                _, prediction = trained_model(
                    item["input_ids"].unsqueeze(dim=0).to(device),
                    item["attention_mask"].unsqueeze(dim=0).to(device)
                )
                # pass the model outputs, which are logits, through softmax function
                prediction = nn.functional.softmax(prediction, dim=1)
                # append predictions to a list to use later
                predictions.append(prediction.flatten())
                labels.append(item["labels"].int())

        # detach from device and ensure move to cpu
        predictions = torch.stack(predictions).detach().cpu()
        labels = torch.stack(labels).detach().cpu()

        # calculate some metrics
        print(f"accruacy for binary classification: {accuracy(predictions, labels)}")

        y_pred = predictions.numpy()
        y_pred_proba = y_pred[:, 1]
        y_true = labels.numpy()
        y_pred_labels = np.argmax(y_pred, axis=1)

        print(f"y pred labels: {y_pred_labels}")

        LABEL_COLUMNS = ["Accepted", "Rejected"]
        print(classification_report(
            y_true,
            y_pred_labels,
            target_names=LABEL_COLUMNS,
            zero_division=0
        ))

        inst_auroc = auroc(predictions, labels, num_classes=2)
        print(f"Instance binary classification auroc: {inst_auroc}")

        # plot roc curve
        thresh = 0.5
        fpr_test, tpr_test, thresholds_test = roc_curve(y_true, y_pred_proba)

        auc_test = roc_auc_score(y_true, y_pred_proba)
        plt.figure(figsize=(12, 8))
        plt.plot(fpr_test, tpr_test, 'g-', label='Test AUC: %.2f' % auc_test)
        plt.plot([0, 1], [0, 1], '-k')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.suptitle("Evaluation - AUC ROC ")
        plt.legend()
        #     plt.savefig(f"{save_dir}/Eval-AUC.png")
        plt.show()

        # plot confusion matrix
        cf = confusion_matrix(y_true, y_pred_labels, normalize='true')
        df_cf = pd.DataFrame(cf, ['accepted', 'rejected'], ['accepted', 'rejected'])
        plt.figure(figsize=(6, 6))
        plt.suptitle("Rejected vs not accepted")
        sns.heatmap(df_cf, annot=True, cmap='Blues')
        #     plt.savefig(f"{save_dir}/Confusion_Matrix.png")
        plt.show()

        print("=" * 50)

# run script
if __name__ == "__main__":
    main()