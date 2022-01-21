from openprompt.data_utils import InputExample
import torch
import pandas as pd
import os
import json, csv
from abc import ABC, abstractmethod
from collections import defaultdict, Counter
from typing import List, Dict, Callable

from openprompt.utils.logging import logger

from openprompt.data_utils.utils import InputExample
from openprompt.data_utils.data_processor import DataProcessor

import pandas as pd
import numpy as np
from tqdm import tqdm

from torchnlp.encoders import LabelEncoder



class MimicProcessor(DataProcessor):
    # TODO Test needed
    def __init__(self):
        super().__init__()        

    def get_examples(self, data_dir, set = "train"):
        path = f"{data_dir}/{set}.csv"
        print(f"loading {set} data")
        print(f"data path provided was: {path}")
        examples = []
        df = pd.read_csv(path)
        self.label_encoder = LabelEncoder(np.unique(df.label).tolist(), reserved_labels = [])
        
        for idx, row in tqdm(df.iterrows()):
#             print(row)
            _, body, label = row
            label = self.label_encoder.encode(label)
#             print(f"body : {body}")
#             print(f"label: {label}")
#             print(f"labels original: {self.label_encoder.index_to_token[label]}")
            
            text_a = body.replace('\\', ' ')

            example = InputExample(
                guid=str(idx), text_a=text_a, label=int(label))
            examples.append(example)
            
        logger.info(f"Returning {len(examples)} samples!")      
        return examples
