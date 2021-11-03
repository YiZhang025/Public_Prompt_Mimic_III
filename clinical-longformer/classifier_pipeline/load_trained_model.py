from numpy.lib.histograms import _histogram_dispatcher
import torch
from transformers import AutoModelForSequenceClassification,AutoTokenizer

from torchnlp.encoders import Encoder
from torchnlp.encoders.text import stack_and_pad_tensors
from torchnlp.encoders.text.text_encoder import TextEncoder
# from tokenizer import Tokenizer
from classifier_one_label import Classifier

import argparse
import os
from datetime import datetime
from pathlib import Path
import bios
import numpy as np


model_dir= "/home/niallt/NLP_DPhil/NLP_Mimic_only/clinical-longformer/experiments/emilyalsentzer/Bio_ClinicalBERT/version_20-09-2021--11-08-38/checkpoints/epoch=3-step=2871.ckpt"
hparams_file = "/home/niallt/NLP_DPhil/NLP_Mimic_only/clinical-longformer/experiments/emilyalsentzer/Bio_ClinicalBERT/version_20-09-2021--11-08-38/hparams.yaml"



# below works - but has to some funky shit

hparams = bios.read(hparams_file)

hparams = argparse.Namespace(**hparams)

model = Classifier(hparams = hparams)

checkpoint = torch.load(model_dir)

model.load_state_dict(checkpoint['state_dict'])

print(model)
