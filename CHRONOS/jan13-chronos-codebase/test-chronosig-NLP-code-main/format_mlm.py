import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import argparse


def dfs_to_list(filenames:list):

    print(f"Reading in files")
    return [pd.read_csv(file) for file in tqdm(filenames)]

def concatenate_all_text(dataframes: list, text_col = "Clinical_Note_Text"):
    '''
    Funciton: Takes in list of dataframes and a column string/identifier. Returns all documented concatenated as big list

    '''
    # get all data from relevant columns
    dfs = dataframes
    # combine all text from the column of interest
    text_concat = pd.concat(dfs, axis=0)["Clinical_Note_Text"]

    print(f"Number of unique text documents is: {len(text_concat)}")
    print(f"The total number of words contained is roughly: {sum([len(elem) for elem in text_concat])}")

    return text_concat


def save_train_val_splits(text_list, save_dir, valid_size=0.33, seed=1234, sample=False):
    # split all text into a training and validation split
    train_data, valid_data = train_test_split(text_list, test_size=valid_size, shuffle=True, random_state=seed)

    if sample:
        print("using sample")
        train_data = train_data[0:10]
        valid_data = valid_data[0:5]
        print(f"length of training data is {len(train_data)}")

    print(f"Saving training data!")
    # save training data
    training_fname = f"{save_dir}/train_mlm.txt"
    train_textfile = open(training_fname, "w", encoding="utf-8")
    for element in tqdm(train_data):
        #         print(element+"\n")
        train_textfile.write(element + "\n")

    train_textfile.close()

    # save validation data
    print(f"Saving validation data!")
    valid_fname = f"{save_dir}/valid_mlm.txt"
    valid_textfile = open(valid_fname, "w", encoding="utf-8")
    for element in tqdm(valid_data):
        valid_textfile.write(element + "\n")
    valid_textfile.close()

    # save all the data

    print("saving all text ")

    all_fname = f"{save_dir}/all_mlm.txt"
    all_textfile = open(all_fname, "w", encoding="utf-8")
    for element in tqdm(text_list):
        all_textfile.write(element + "\n")
    all_textfile.close()


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--data_dir",
                        default = "F:/OxfordTempProjects/PatientTriageNLP/",
                        type=str,
                        help = "The data path to the directory containing the notes and referral data files")

    parser.add_argument("--data_files",
                        default=["F:/OxfordTempProjects/PatientTriageNLP/processed_data/processed_pre_referral_long.csv",
                                 "F:/OxfordTempProjects/PatientTriageNLP/processed_data/processed_post_referral_long.csv",
                                 "F:/OxfordTempProjects/PatientTriageNLP/processed_data/preprocessed_pre14_post28_referral_notes.csv"],
                        type=list,
                        help="The data path to the directory containing the notes and referral data files")

    parser.add_argument("--text_col",
                        default = "Clinical_Note_Text",
                        type=str,
                        help = "The data path to the directory containing the notes and referral data files")

    parser.add_argument("--save_dir",
                        default = "F:/OxfordTempProjects/PatientTriageNLP/processed_data/text_MLM",
                        type=str,
                        help = "The data path to the directory containing the notes and referral data files",
                        required = True)


    args = parser.parse_args()

    # read provided data files and store in a list for later concatenation
    dataframes = dfs_to_list(filenames = args.data_files)
    # concat all text columns into one big list
    all_text = concatenate_all_text(dataframes, args.text_col)
    #split into training and validation sets and save to file
    save_train_val_splits(all_text, args.save_dir, valid_size=0.33, seed=1234, sample=False)

# run script
if __name__ == "__main__":
    main()

