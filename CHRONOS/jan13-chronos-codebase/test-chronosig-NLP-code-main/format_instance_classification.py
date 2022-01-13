import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from loguru import logger
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit
import argparse

'''
Script to take cleaned pre-referral-notes data, derive reject/accept labels and split into training/test/val splits


example usage: python format_instance_classification.py --balance_data --group_ids 
'''


def get_rejection_labels(episode_days, threshold=28):

    '''
    Function to derive accept/reject labels based on episode days
    :param episode_days: Int - number of days between referral and either discharge date or none
    :param threshold: Int - number of days to decide whether patient was accepted or rejected. i.e. episode days < threshold = reject
    :return: Int - applied to dataframe, 0 (accept) or 1 (reject)
    '''

    if episode_days < threshold:
        return 1
    else:
        return 0

def write_to_file(df, path):
    print(f"Saving following: {path}")
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    return df.to_csv(path, index=False)

def create_training_test_data(long_data_file = None, concat_data_file = None, save_dir = None,
                              train_size=0.8, test_size = 0.5, seed=42, threshold = 28,
                              shuffle = False, label_col = "rejection",group_ids = True,balanced_data = True):

    '''
    Function to create training/test splits - will expect both long form and concat documents now - this allows sensible grouping.
    :param long_data_file: String - path to clinical documents in long form i.e. kept as they are, one per row
    :param concat_data_file: String - path to clinical documents in concatenated form i.e. concatenated based on BRC_ID/instance number
    :param save_dir: String - path to save training/test data derived from long form and concat data. These will be altered within function
    :param train_size:Int - prportion to split data into training/test
    :param test_size: Int - proportion to split test into val/test
    :param seed: Int - random seed - allows reproducability
    :param threshold: Int - threshold to be passed to rejection label derivation
    :param shuffle: Bool - whether or not to shuffle the data when splitting
    :param label_col: String - Name of column containing clinical text, i.e. X data for training
    :param group_ids: Bool - whether or not to group the data by ids
    :param balanced_data: Bool - whether or not to balance the data based on minority class distribution
    :return: training/test/val data splits for both long and concatenated data - saved as provided paths
    '''
    # read in long data i.e.long_data_file
    df = pd.read_csv(long_data_file)

    # get the labels
    logger.warning(f"Deriving rejection labels based on threshold of :{threshold} days")
    df[label_col] = df["episode_days"].apply(get_rejection_labels, threshold)

    # read in concat data file
    df_concat = pd.read_csv(concat_data_file)
    df_concat[label_col] = df_concat["episode_days"].apply(get_rejection_labels, threshold)


    # balance the data to the create a sample with roughly equal class numbers - based on number of minority class samples
    if balanced_data:

        # first resample the long form data
        least_samples = df[label_col].value_counts().min()
        logger.warning(f"Balancing long dataset by subsampling the majority class to match the number of samples in minority class. Resample size is:\n {least_samples}")
        df = df.groupby(label_col).sample(least_samples, random_state = seed).copy()
        logger.info(f"Class distribution now: {df[label_col].value_counts()}")


        # now do same for concatenated
        least_samples_concat = df_concat[label_col].value_counts().min()
        logger.warning(f"Balancing dataset concat by subsampling the majority class to match the number of samples in minority class. Resample size is:\n {least_samples_concat}")
        df_concat = df_concat.groupby(label_col).sample(least_samples_concat, random_state = seed).copy()
        logger.info(f"Class distribution now:\n {df_concat[label_col].value_counts()}")
    # use grouped shuffle split to split training and test sets, ensuring no training data patients are in test sets
    # bit crude but we do this twice to create training and test - then again to create test and val
    if group_ids:
        logger.warning("Grouping the data based on BRC_IDS. Meaning the training and test sets will not share any participants")
        train_gss = GroupShuffleSplit(n_splits=1, train_size=train_size, random_state=seed)


        for train_idx, rem_idx in train_gss.split(df, groups=df.BRC_ID):
            logger.warning("getting training and test indices for spliting LONG data")

        # now set the training and test datasets with these new grouped indices
        # train data
        train_data = df.iloc[train_idx]
        # remaining data for further splitting
        rem_data = df.iloc[rem_idx]

        # assert that there is no intersection BRC_IDS i.e. BRC_IDS in test are not also in remaining data
        assert len(set(train_data.BRC_ID.unique()).intersection(set(rem_data.BRC_ID.unique()))) is 0


        # now apply same logic to split the remaining data into a validation and holdout test set
        val_gss = GroupShuffleSplit(n_splits=1, train_size=test_size, random_state=seed)


        for test_idx, val_idx in val_gss.split(rem_data, groups=rem_data.BRC_ID):
            logger.warning("getting test and validation indices for splitting LONG data")

        # now set the training and test datasets with these new grouped indices
        test_data = rem_data.iloc[test_idx]
        valid_data = rem_data.iloc[val_idx]

        # assert that there is no intersection BRC_IDS i.e. BRC_IDS in test are not also in remaining data
        assert len(set(test_data.BRC_ID.unique()).intersection(set(valid_data.BRC_ID.unique()))) is 0

        # now we use the same group shuffler on the concat data - should mean both training and test sets contain same brc_ids
        for train_idx_concat, rem_idx_concat in train_gss.split(df_concat, groups=df_concat.BRC_ID):
            logger.warning("getting training and test indices for split of CONCAT data")

        # now set the training and test datasets with these new grouped indices
        train_data_concat = df_concat.iloc[train_idx_concat]
        rem_data_concat = df_concat.iloc[rem_idx_concat]
        # assert that there is no intersection BRC_IDS i.e. BRC_IDS in test are not also in remaining data
        assert len(set(train_data_concat.BRC_ID.unique()).intersection(set(rem_data_concat.BRC_ID.unique()))) is 0

        for test_idx_concat, val_idx_concat in val_gss.split(rem_data_concat, groups=rem_data_concat.BRC_ID):
            logger.warning("getting test and validation indices for spliting CONCAT data")

        # now set the training and test datasets with these new grouped indices
        test_data_concat = rem_data_concat.iloc[test_idx_concat]
        valid_data_concat = rem_data_concat.iloc[val_idx_concat]

        # assert that there is no intersection BRC_IDS i.e. BRC_IDS in test are not also in remaining data
        assert len(set(test_data_concat.BRC_ID.unique()).intersection(set(valid_data_concat.BRC_ID.unique()))) is 0
    ########################################################################################################
    # # OLD FUNCTION TO  create data splits - following will create a 80-10-10 split
    # train_data, rem_data = train_test_split(df, train_size = train_size, shuffle=shuffle, random_state=seed)
    #
    # valid_data, test_data = train_test_split(rem_data, test_size = test_size, shuffle=shuffle, random_state=seed)
    ########################################################################################################

    # write long form data to file
    write_to_file(train_data, f"{save_dir}/long/train.csv")
    write_to_file(valid_data, f"{save_dir}/long/valid.csv")
    write_to_file(test_data, f"{save_dir}/long/test.csv")

    # write concatenated data to file
    write_to_file(train_data_concat, f"{save_dir}/concat/train.csv")
    write_to_file(valid_data_concat, f"{save_dir}/concat/valid.csv")
    write_to_file(test_data_concat, f"{save_dir}/concat/test.csv")

def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--data_dir",
                        default = "F:/OxfordTempProjects/PatientTriageNLP/",
                        type=str,
                        help = "The data path to the directory containing the notes and referral data files")

    parser.add_argument("--long_data_file",
                        default="F:/OxfordTempProjects/PatientTriageNLP/processed_data/processed_pre_referral_instances_long.csv",
                        type=str,
                        help="The data path to the directory containing the notes and referral data files")

    parser.add_argument("--concat_data_file",
                        default="F:/OxfordTempProjects/PatientTriageNLP/processed_data/processed_pre_referral_instances_concat.csv",
                        type=str,
                        help="The data path to the directory containing the notes and referral data files")

    parser.add_argument("--text_col",
                        default = "Clinical_Note_Text",
                        type=str,
                        help = "The data path to the directory containing the notes and referral data files")

    parser.add_argument("--save_dir",
                        default = "F:/OxfordTempProjects/PatientTriageNLP/processed_data/instance_classification/",
                        type=str,
                        help = "The data path to the directory containing the notes and referral data files"
                        )

    parser.add_argument("--balance_data",
                        action = 'store_true',
                        help="Whether not to balance dataset based on least sampled class")

    parser.add_argument("--group_ids",
                        action = 'store_true',
                        help="Whether not to group the train/test/val sets based on BRC_ID")

    parser.add_argument("--randomize",
                        action = 'store_true',
                        help="Whether not to randomly permutate the Y labels to create a randomised dataset for null model training")

    parser.add_argument("--shuffle",
                        action = 'store_true',
                        help="Whether not to shuffle the data rows to ensure no order is left by grouping functions")

    args = parser.parse_args()


    #set up parameters
    save_dir = Path(args.save_dir)

    if args.balance_data:
        logger.warning("Will be balancing the data based on minority class")
        balance_data = True
        save_dir = save_dir /"balanced"

    else:
        logger.warning("Will NOT be balancing the data based on minority class. Expect imbalanced datasets!")
        balance_data = False
        save_dir = save_dir /"unbalanced"

    if args.group_ids:
        group_ids = True
        # save_dir = save_dir / "grouped_id"
    else:
        group_ids = False


    if args.shuffle:
        shuffle = True
    else:
        shuffle = False
    # run the create_training.... function with arguments provided
    #TODO add all arguments to the parser
    create_training_test_data(long_data_file=args.long_data_file, concat_data_file=args.concat_data_file, save_dir = save_dir,
                                          train_size=0.8, test_size = 0.5, seed=42, threshold = 28,
                                                shuffle=shuffle, group_ids=group_ids, balanced_data=balance_data)

# run script
if __name__ == "__main__":
    main()

