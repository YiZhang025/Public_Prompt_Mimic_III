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


#TODO - may need to refactor how labe encoder is instantiated. At moment it does it separatley for each set

class Mimic_ICD9_Processor(DataProcessor):


    '''
    Function to convert mimic icd9 dataset to a open prompt ready dataset. 
    
    We also instantiate a LabelEncoder() class which is fitted to the given dataset. Fortunately it appears
    to create the same mapping for each set, given each set contains all classes. 

    This is not ideal, and need to think of a better way to store the label encoder based on training data.
    

  
    
    '''
    # TODO Test needed
    def __init__(self):
        super().__init__()        

    def get_examples(self, data_dir, mode = "train", label_encoder = None,
                     generate_class_labels = False, class_labels_save_dir = "scripts/mimic_icd9_top50/"):

        path = f"{data_dir}/{mode}.csv"
        print(f"loading {mode} data")
        print(f"data path provided was: {path}")
        examples = []
        df = pd.read_csv(path)

        # need to either initializer and fit the label encoder if not provided
        if label_encoder is None:
            self.label_encoder = LabelEncoder(np.unique(df.label).tolist(), reserved_labels = [])
        else: 
            print("we were given a label encoder")
            self.label_encoder = label_encoder

        
        for idx, row in tqdm(df.iterrows()):
#             print(row)
            body, label = row['text'],row['label']
            label = self.label_encoder.encode(label)
#             print(f"body : {body}")
#             print(f"label: {label}")
#             print(f"labels original: {self.label_encoder.index_to_token[label]}")
            
            text_a = body.replace('\\', ' ')

            example = InputExample(
                guid=str(idx), text_a=text_a, label=int(label))
            examples.append(example)
            
        logger.info(f"Returning {len(examples)} samples!") 

#         now we want to return a list of the non-encoded labels based on the fitted label encoder
        if generate_class_labels:
            logger.info(f"Saving class labels to: {class_labels_save_dir}")
            class_labels = self.generate_class_labels()
            # write these to files as the classes for prompt learning pipeline

            textfile = open(f"{class_labels_save_dir}/labels.txt", "w")

            # write each label to a file separated by new line, but do not add new line to last entry as this will create an empty "" label
            for element in class_labels[:-1]:

                textfile.write(element + "\n")
            # now write the last item to the file
            textfile.write(class_labels[-1])
            textfile.close() 

        return examples

    def generate_class_labels(self):
        # now we want to return a list of the non-encoded labels based on the fitted label encoder
        try:
            return list(self.label_encoder.tokens.keys())
        except:
            print("No class labels as haven't fitted any data yet. Run get_examples first!")
            raise NotImplementedError

    
    def load_class_labels(self, file_path = "./scripts/mimic_icd9_top50/labels.txt"):
        # function to load pre-generated class labels
        # returns list of class labels

        text_file = open(f"{file_path}", "r")

        class_labels = text_file.read().split("\n")

        return class_labels




class Mimic_ICD9_Triage_Processor(DataProcessor):


    '''
    Function to convert mimic icd9 triage dataset to a open prompt ready dataset. 
    
    We also instantiate a LabelEncoder() class which is fitted to the given dataset. Fortunately it appears
    to create the same mapping for each set, given each set contains all classes. 

    This is not ideal, and need to think of a better way to store the label encoder based on training data.
    

  
    
    '''
    # TODO Test needed
    def __init__(self):
        super().__init__()        

    def get_examples(self, data_dir, mode = "train", label_encoder = None,
                     generate_class_labels = False, class_labels_save_dir = "./scripts/mimic_triage/"):

        path = f"{data_dir}/{mode}.csv"
        print(f"loading {mode} data")
        print(f"data path provided was: {path}")
        examples = []
        df = pd.read_csv(path)


        # need to either initializer and fit the label encoder if not provided
        if label_encoder is None:
            self.label_encoder = LabelEncoder(np.unique(df["triage-category"]).tolist(), reserved_labels = [])
        else: 
            print("we were given a label encoder")
            self.label_encoder = label_encoder

        
        for idx, row in tqdm(df.iterrows()):
#             print(row)
            body, label = row['text'],row['triage-category']
            label = self.label_encoder.encode(label)
#             print(f"body : {body}")
#             print(f"label: {label}")
#             print(f"labels original: {self.label_encoder.index_to_token[label]}")
            
            text_a = body.replace('\\', ' ')

            example = InputExample(
                guid=str(idx), text_a=text_a, label=int(label))
            examples.append(example)
            
        logger.info(f"Returning {len(examples)} samples!") 

#         now we want to return a list of the non-encoded labels based on the fitted label encoder
        if generate_class_labels:
        
            if not os.path.exists(class_labels_save_dir):
                os.makedirs(class_labels_save_dir)
            logger.info(f"Saving class labels to: {class_labels_save_dir}")
            class_labels = self.generate_class_labels()
            # write these to files as the classes for prompt learning pipeline           

            textfile = open(f"{class_labels_save_dir}/labels.txt", "w")

            for element in class_labels[:-1]:

                textfile.write(element + "\n")
                # now write the last item to the file
            textfile.write(class_labels[-1])
            textfile.close() 

        return examples

    def generate_class_labels(self):
        # now we want to return a list of the non-encoded labels based on the fitted label encoder
        try:
            return list(self.label_encoder.tokens.keys())
        except:
            print("No class labels as haven't fitted any data yet. Run get_examples first!")
            raise NotImplementedError

    
    def load_class_labels(self, file_path = "./scripts/mimic_triage/labels.txt"):
        # function to load pre-generated class labels
        # returns list of class labels

        text_file = open(f"{file_path}", "r")

        class_labels = text_file.read().split("\n")

        return class_labels


class Mimic_Mortality_Processor(DataProcessor):


    '''
    Function to convert mimic mortality prediction dataset from the clinical outcomes paper: https://aclanthology.org/2021.eacl-main.75/
    
    to a open prompt ready dataset. 
    
    We also instantiate a LabelEncoder() class which is fitted to the given dataset. Fortunately it appears
    to create the same mapping for each set, given each set contains all classes.    
    
    '''
    # TODO Test needed
    def __init__(self):
        super().__init__()        

    def get_examples(self, data_dir, mode = "train", label_encoder = None,
                     generate_class_labels = False, class_labels_save_dir = "./scripts/mimic_mortality/"):

        path = f"{data_dir}/{mode}.csv"
        print(f"loading {mode} data")
        print(f"data path provided was: {path}")
        examples = []
        df = pd.read_csv(path)

        # map the binary classification label to a new string class label
        df["label"] = df["hospital_expire_flag"].map({0:"alive",1:"deceased"})
        
        # need to either initializer and fit the label encoder if not provided
        if label_encoder is None:
            self.label_encoder = LabelEncoder(np.unique(df["label"]).tolist())
        else: 
            print("we were given a label encoder")
            self.label_encoder = label_encoder

        
        for idx, row in tqdm(df.iterrows()):
#             print(row)
            body, label = row['text'],row['label']
            label = self.label_encoder.encode(label)
#             print(f"body : {body}")
#             print(f"label: {label}")
#             print(f"labels original: {self.label_encoder.index_to_token[label]}")
            
            text_a = body.replace('\\', ' ')

            example = InputExample(
                guid=str(idx), text_a=text_a, label=int(label))
            examples.append(example)
            
        logger.info(f"Returning {len(examples)} samples!") 

#         now we want to return a list of the non-encoded labels based on the fitted label encoder
        if generate_class_labels:
        
            if not os.path.exists(class_labels_save_dir):
                os.makedirs(class_labels_save_dir)
            logger.info(f"Saving class labels to: {class_labels_save_dir}")
            class_labels = self.generate_class_labels()
            # write these to files as the classes for prompt learning pipeline           

            textfile = open(f"{class_labels_save_dir}/labels.txt", "w")

            for element in class_labels[:-1]:

                textfile.write(element + "\n")
            # now write the last item to the file
            textfile.write(class_labels[-1])
            textfile.close() 

        return examples

    def generate_class_labels(self):
        # now we want to return a list of the non-encoded labels based on the fitted label encoder
        try:
            return list(self.label_encoder.tokens.keys())
        except:
            print("No class labels as haven't fitted any data yet. Run get_examples first!")
            raise NotImplementedError

    
    def load_class_labels(self, file_path = "./scripts/mimic_mortality/labels.txt"):
        # function to load pre-generated class labels
        # returns list of class labels

        text_file = open(f"{file_path}", "r")

        class_labels = text_file.read().split("\n")

        return class_labels