
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import re
import string
from nltk.tokenize import sent_tokenize, word_tokenize
import random
import spacy
import argparse
from datetime import datetime, timedelta

from loguru import logger



"""
Script and dataclass to handle the typical ingestion, cleaning, feature  and engineering required for referral instance derived tasks.

Requires:


Example usage from cmd line: python clean_and_preprocess_data.py --data_dir F:/OxfordTempProjects/PatientTriageNLP/ --

"""

class TextData:
    def __init__(self, data_dir, patient_cohort_path,
                 pre_referral_data_path,
                 post_referral_data_path,
                 pre14_prior28_attachments_path,
                 save_dir, admin_language,
                 sample,sample_size):
        self.data_dir = data_dir
        self.save_dir = save_dir
        self.patient_cohort_path = patient_cohort_path
        self.pre_referral_data_path = pre_referral_data_path
        self.post_referral_data_path = post_referral_data_path
        self.pre14_prior28_attachments_path = pre14_prior28_attachments_path
        self.admin_language = admin_language
        self.sample = sample
        self.sample_size = sample_size
        # cannot use nltk tokenizer at moment  - as needs to download
        self.tokenizer = word_tokenize
        self.read_data()

    def read_data(self):

        # read preprocessed data files
        # IMPORTANT - specify the date column to be read in as datetime object
        self.referral_data = pd.read_excel(f"{self.data_dir}{self.patient_cohort_path}")
        # have to crudely convert discharge date
        self.referral_data["Discharge_Date"] = pd.to_datetime(self.referral_data["Discharge_Date"], errors = 'coerce')

        self.pre_referral_data = pd.read_csv(f"{self.data_dir}{self.pre_referral_data_path}",
                                         parse_dates=["Clinical_Note_Date"])
        self.post_referral_data = pd.read_csv(f"{self.data_dir}{self.post_referral_data_path}",
                                            parse_dates=["Clinical_Note_Date"])
        self.pre14_post28_attachments_data = pd.read_csv(f"{self.data_dir}{self.pre14_prior28_attachments_path}",
                                                         parse_dates=["Doc_Date"])
        self.pre14_post28_attachments_data.rename({"Doc_Date": "Clinical_Note_Date",
                                         "Attachment_File_Body": "Clinical_Note_Text",
                                         "General_Document_Category_Value": "Clinical_Note_Type_Value"}, axis=1,
                                        inplace=True)


    def clean_data(self,admin_language,
                   col_name,
                   notes_df: pd.DataFrame,
                   min_tokens = 10,
                   note_type_drop = ["\\N", "72 Hours follow up",
                                   "COVID-19 Information", "Living Will",
                                   "Consent Form",
                                   "Discharge Summary (External)", "Section 17"]
                   ) -> pd.DataFrame:

            """
            remove redundant information from free text using some common NLP techniques
                and heuristic rules

            Args:
                col_name (string): name of column with text in
                notes_df (pd.DataFrame): CRIS produced clinical notes with following possible cols:
                ['BRC_ID', 'Document_ID', 'Clinical_Note_Date',
               'Clinical_Note_Type_Value', 'Clinical_Note_Category_Value',
               'Clinical_Note_Text']

           Returns: pd.DataFrame: notes_df, filtered of redundant text
            """
            filtered_df = notes_df.copy()

            # remove rows with no data/clinical date data

            filtered_df = filtered_df[filtered_df["Clinical_Note_Date"] != "\\N"]

            # convert datetime column to pd.datetime
            filtered_df["Clinical_Note_Date"] = pd.to_datetime(filtered_df["Clinical_Note_Date"]).dt.date

            # remove some punctuation
            filtered_df[col_name] = filtered_df[col_name].replace(r"\[.*?\]", "", regex=True)

            logger.info("Cleaning and applying some regex rules!")
            # remove refundnant new lines etc
            for original, replacement in tqdm([
                ("\n", " "),
                ("\r\n\r", " "),
                ("\r", " "),
                ("\t"," "),
                ("w/", "with"),
                ("_", " "),
                ("-", " "),
                ("*", ' '),
                ("  ", " "),
                ('"', '' )
            ]):
                filtered_df[col_name] = filtered_df[col_name].str.replace(original, replacement)
            # strip white space and lower case
            filtered_df[col_name] = filtered_df[col_name].str.strip()
            filtered_df[col_name] = filtered_df[col_name].str.lower()

            # found white spaces still persist - so now double remove them
            filtered_df[col_name] = [re.sub(r"\s+",' ',x) for x in filtered_df[col_name]]


            # Try remove some common admin language items
            for admin_token in admin_language:
                # print(f"removing following token/passage :{admin_token}")
                filtered_df[col_name] = filtered_df[col_name].str.replace(admin_token, " ")
            # double check for double spaces
            filtered_df[col_name] = filtered_df[col_name].str.replace("  ", " ")
            # cannot load nltk or spacy tokenizers through restricted internet at moment
            filtered_df["tokenized_text"] = self.basic_tokenize(filtered_df, col_name)

            # count num tokens
            filtered_df["num_tokens"] = filtered_df["tokenized_text"].apply(lambda x: len(x))
            # some empty strings may be left over - replace empty with nan
            filtered_df.replace("", np.nan, inplace=True)
            # subset rows with greater than min tokens
            filtered_df = filtered_df[filtered_df["num_tokens"]>min_tokens]

            # drop certain document types
            # drop certain pre-referral document types
            filtered_df = filtered_df.query("Clinical_Note_Type_Value not in @note_type_drop")

            # drop na

            return filtered_df.dropna()

    def compare_random_original_filtered(self, org_df, filt_df, col_name):
        sample_id = random.randint(0, len(org_df))
        print(f"original clinical text: {org_df[col_name][sample_id]} \n")
        print("=" * 50)
        print(f"filtered clinical text: {filt_df[col_name][sample_id]}")

    def basic_tokenize(self, filtered_notes_df, col_name):
        return filtered_notes_df[col_name].apply(lambda x: x.split())

    def get_referral_instance(self,clinical_note_date, instance_count, date, running_min_date):
        """
        Function to use inside of lambda_count_referrals_instance. Takes in clinical_note_date column
        and returns the instance count number dependent on whether it is newer than the current running min date

        args:
            clinical_note_date (datetime): column containing the date of the clinical text document/note
            instance_count (number/int): the current referral instance count
            date (datetime): the referral date used as upper bound for an instance membership
            running_min_date (datetime): the moving lower bound date to determine instance membership

        """

        if (clinical_note_date <= date) & (clinical_note_date > running_min_date):

            return instance_count
        else:
            #        if the clinical note date is not above the running_min_date - then it will be instance 0
            return 0

    def count_referrals_instances(self, notes_data, referral_data, lookback_window = 0):

        """

        """

        sample = self.sample
        sample_size = self.sample_size
        # get all unique brc_ids - if sample is True - grab sample size
        if sample:
            print("Sample has been selected. Sampling based on sample size:  ", sample_size)
            try:
                brc_ids = notes_data.BRC_ID.unique()[0:sample_size]
                print(f" collecting data for following IDS: {brc_ids}")
            except:
                print(f"no BRC_ID column found in notes_data provided. The columns were: {notes_data.columns}")
        else:
            try:
                print(f"getting all BRC ids")
                brc_ids = notes_data.BRC_ID.unique()
            except:
                print(f"no BRC_ID column found in notes_data provided. The columns were: {notes_data.columns}")

        # drop all notes with a type that is contained within note_type_drop argument

        # empty list to fill with brc_id specific dataframes
        new_dfs = []

        logger.info(f"Beginning the acquisition of referral instances for {len(brc_ids)} users data!")
        for brc_id in tqdm(brc_ids):
            # get dataframe based on BRC_ID
            patient_referral_data = referral_data[referral_data["BRC_ID"] == brc_id].copy()

            patient_notes_df = notes_data[notes_data["BRC_ID"] == brc_id].copy()
            # add initial referral instance variable to be changed
            patient_notes_df["referral_instance"] = 0
            # add zero column for referral location

            # get list of unique dates and sort em
            referral_dates = patient_referral_data["Referral_Date"].sort_values(ascending=True)

            # get the minimum and max referral date for that brc_id
            min_referral = referral_dates.min()
            max_referral = referral_dates.max()

            min_clinical_note_date = patient_notes_df["Clinical_Note_Date"].min()

            # set initial referral instance count  = 0
            instance_counter = 0
            # initialise minimum date as the minimum referral date recorded
            running_min_date = min_referral

            # set up dictionaries for mapping against date values
            instance_date_dict = {}
            instance_location_dict = {}
            instance_discharge_date_dict = {}
            for date in referral_dates:
                # for each referral date - find documents related to that instance
                instance_date_dict[instance_counter] = date
                # this can then be used to map the locations
                instance_location_dict[instance_counter] = \
                patient_referral_data[patient_referral_data["Referral_Date"] == date]["Location_Name"].values[0]


                instance_discharge_date_dict[instance_counter] = \
                patient_referral_data[patient_referral_data["Referral_Date"] == date]["Discharge_Date"].values[0]

                # I think this works, but because some documents end up not being
                change_idx = patient_notes_df.index[(patient_notes_df["Clinical_Note_Date"] <= date) &
                                                    (patient_notes_df["Clinical_Note_Date"] > running_min_date)]

                #             print(f"change idx: {change_idx} of length : {len(change_idx)}")
                #             print(f"current instance counter : {instance_counter}")

                #             print(f"data relevant to this counter: {patient_notes_df.loc[change_idx]}")

                patient_notes_df.loc[change_idx, "referral_instance"] = instance_counter

                running_min_date = date

                #  HOT FIX- at moment some referral dates, have no pre-referral documents linked. If this happens - don't increase counter

                if len(change_idx) > 0:
                    instance_counter += 1

            #         add a variable: number of referral instances i.e. max of referral instance +1
            patient_notes_df["number_referral_instances"] = patient_notes_df["referral_instance"].max() + 1

            # add referral date
            patient_notes_df["referral_date"] = pd.to_datetime(
                patient_notes_df["referral_instance"].map(instance_date_dict))

            # add the referral location
            patient_notes_df["referral_location"] = patient_notes_df["referral_instance"].map(instance_location_dict)
            # add the discharge date
            patient_notes_df["discharge_date"] = pd.to_datetime(
                patient_notes_df["referral_instance"].map(instance_discharge_date_dict))

            # calculate days between aka episode days
            patient_notes_df["episode_days"] = (
                        patient_notes_df["discharge_date"] - patient_notes_df["referral_date"]).dt.days

            # HOT FIX - force clinical note date to be a datetime object
            patient_notes_df["Clinical_Note_Date"] = pd.to_datetime(patient_notes_df["Clinical_Note_Date"])

            # remove any notes that are too far away from referral date, based on lookbackwindows
            if lookback_window >0:
                min_lookback_date = (
                            pd.to_datetime(patient_notes_df["referral_instance"].map(instance_date_dict)) - timedelta(
                        days=lookback_window)).values[0]

                patient_notes_df = patient_notes_df[patient_notes_df["Clinical_Note_Date"] >= min_lookback_date]

            new_dfs.append(patient_notes_df)

        # now combine all of the brc_id specific dataframes into one - concatenate/stack
        all_data_merged = pd.concat(new_dfs, axis=0)
        all_data_merged.reset_index(inplace=True, drop=True)

        print(f"Final data to be returned is of shape: {all_data_merged.shape}")

        return all_data_merged


    # function to concatenate all text per instance per row - each row becomes a unique instance with all data attached
    def concatenate_instance_documents_long(self,df_w_instances, referral_instance_col = "referral_instance",
                                            referral_number_col = "number_referral_instances", text_col = "Clinical_Note_Text",
                                            sample=False,
                                            sample_size=5,
                                            chronological = False):

        logger.warning("About to concatenate unique instance texts together!")
        print("="*50)
        notes_data = df_w_instances.copy()
        # get brc_ids
        # get all unique brc_ids - if sample is True - grab sample size
        if sample:
            print("Sample has been selected. Sampling based on sample size:  ", sample_size)
            try:
                brc_ids = notes_data.BRC_ID.unique()[0:sample_size]
                print(f" collecting data for following IDS: {brc_ids}")
            except:
                print(f"no BRC_ID column found in notes_data provided. The columns were: {notes_data.columns}")
        else:

            brc_ids = notes_data.BRC_ID.unique()

        instance_dfs = []
        for brc_id in tqdm(brc_ids):
            patient_df = notes_data[notes_data["BRC_ID"] == brc_id].copy()

            #         print(f"patient df is: {patient_df}")

            # use the referral number to create a range to iterate
            num_instances = patient_df[referral_number_col].max()
            #         print(f"number instances = {num_instances}")

            # for each instance loop through and collect the notes/documents leading upto that date
            for instance in np.arange(num_instances):
                # empty dict to fill with key/value pairs
                instance_dict = {}
                # if we want to concatenate documents in chronological order i.e. oldest to newest
                if chronological:
                    patient_df = patient_df.sort_values(by="Clinical_Note_Date", ascending=True)

                else:
                    # we actually want to sort clinical document dates in descending order - so first rows are most recent
                    patient_df = patient_df.sort_values(by="Clinical_Note_Date", ascending=False)

                # concatenates all individual documents and combines into one big ass string
                instance_dict["Clinical_Note_Text"] = ' '.join(
                    patient_df[patient_df[referral_instance_col] == instance][text_col].tolist())
                # create a dataframe from this dictionary
                instance_df = pd.DataFrame([instance_dict])

                # store a few key variables
                instance_df["all_text_length"] = len(instance_df["Clinical_Note_Text"].values[0].split(" "))
                instance_df["BRC_ID"] = brc_id
                instance_df[referral_instance_col] = instance
                instance_df["num_referral_instances"] = num_instances
                instance_df["referral_date"] = \
                patient_df[patient_df[referral_instance_col] == instance]["referral_date"].values[0]
                instance_df["discharge_date"] = \
                patient_df[patient_df[referral_instance_col] == instance]["discharge_date"].values[0]
                instance_df["episode_days"] = \
                patient_df[patient_df[referral_instance_col] == instance]["episode_days"].values[0]

                # append patient dataframe  to list which will contain all patients
                instance_dfs.append(instance_df)

        # now we have a dataframe for each BRC_ID with a list of all documents associated to each referral instance

        # combine all into a single dataframe
        all_data = pd.concat(instance_dfs, axis=0)
        # set_index to the BRC_ID for more readable view
        all_data.set_index("BRC_ID", inplace=True)

        #apply the label derivation?

        logger.info(f"The shape of the long form dataframe is: {all_data.shape}")

        # reset_index to leave BRC_ID as its own column
        return all_data.reset_index()


    # function to re-organise data to have one line per subject with text referral instance related documents concatenated?

    def concatenate_instance_document_subject(self,df_w_instances, referral_instance_col = "referral_instance",
                                      referral_number_col = "number_referral_instances", text_col = "Clinical_Note_Text"):
        notes_data = df_w_instances.copy()
        # get brc_ids
        # get all unique brc_ids - if sample is True - grab sample size
        if self.sample:
            print("Sample has been selected. Sampling based on sample size:  ", self.sample_size)
            try:
                brc_ids = notes_data.BRC_ID.unique()[0:self.sample_size]
                print(f" collecting data for following IDS: {brc_ids}")
            except:
                print(f"no BRC_ID column found in notes_data provided. The columns were: {notes_data.columns}")
        else:

            brc_ids = notes_data.BRC_ID.unique()

        dfs = []

        for brc_id in tqdm(brc_ids):
            patient_df = notes_data[notes_data["BRC_ID"] == brc_id].copy()

            # use the referral number to create a range to iterate
            num_instances = patient_df[referral_number_col].max()
            #         print(f"number instances = {num_instances}")
            patient_dict = {}
            for instance in np.arange(num_instances):
                # TODO - change logic to include all text up until that instance number
                patient_dict[f"instance_{instance}"] = patient_df[patient_df[referral_instance_col] == instance][
                    text_col].tolist()

            df = pd.DataFrame([patient_dict])
            df["BRC_ID"] = brc_id
            df["num_referral_instances"] = num_instances
            dfs.append(df)

        # now we have a dataframe for each BRC_ID with a list of all documents associated to each referral instance

        # combine all into a single dataframe
        all_data = pd.concat(dfs, axis=0)
        # set_index to the BRC_ID for more readable view
        all_data.set_index("BRC_ID", inplace=True)
        logger.info(f"The shape of the returned concatenated by subject dataframe is: {all_data.shape}")

        # reset_index to leave BRC_ID as its own column
        return all_data.reset_index()

    def get_rejection_labels(self,episode_days, threshold=28):
        if episode_days < threshold:
            return 1
        else:
            return 0

    def write_filtered_to_file(self, filtered_df, path):
        print(f"Saving following: {path}")
        directory = os.path.dirname(path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        return filtered_df.to_csv(path, index=False)


    def clean_save_data(self,text_col = "Clinical_Note_Text", lookback_window = 0, chronological = False):
        self.text_column = text_col
        self.lookback_window = lookback_window
        self.chronological = chronological
        # first get the raw data cleaned

        self.cleaned_pre_referral_data = self.clean_data(self.admin_language,self.text_column,self.pre_referral_data)
        self.cleaned_post_referral_data = self.clean_data(self.admin_language,self.text_column,self.post_referral_data)
        self.cleaned_pre14_post28_data = self.clean_data(self.admin_language,self.text_column,self.pre14_post28_attachments_data)

        # now create instance number/counter variables and save to file
        #TODO check if the count_referral_instances works properly for post_referral data/documents
        self.processed_pre_referral_data = self.count_referrals_instances( self.cleaned_pre_referral_data, self.referral_data , self.lookback_window)
        self.processed_pre_referral_docs = self.count_referrals_instances( self.cleaned_pre14_post28_data, self.referral_data , self.lookback_window)
        # self.processed_post_referral_data = self.count_referrals_instances( self.cleaned_post_referral_data, self.referral_data, self.lookback_window)
        # group this data by BRC_ID and concatenate instance documents
        self.grouped_processed_pre_referral_data = self.concatenate_instance_document_subject(self.processed_pre_referral_data)
        # for now not bothering with post_referral
        # self.grouped_processed_post_referral_data = self.concatenate_instance_document(self.processed_post_referral_data)

        # now group by unique instance - so each row is a uniqe instance with all text etc
        self.instance_concat_pre_referral_data = self.concatenate_instance_documents_long(self.processed_pre_referral_data, chronological=self.chronological)
        self.instance_concat_pre_referral_docs = self.concatenate_instance_documents_long(self.processed_pre_referral_docs, chronological=self.chronological)

        # write to file
        ## cleaned original files
        self.write_filtered_to_file(self.cleaned_pre_referral_data, f"{self.save_dir}/processed_pre_referral_notes.csv")
        self.write_filtered_to_file(self.cleaned_post_referral_data, f"{self.save_dir}/processed_post_referral_notes.csv")
        self.write_filtered_to_file(self.cleaned_pre14_post28_data, f"{self.save_dir}/processed_pre14_post28_referral_docs.csv")


        # long form processed data
        self.write_filtered_to_file(self.processed_pre_referral_data,f"{self.save_dir}/processed_pre_referral_instances_long.csv")
        self.write_filtered_to_file(self.processed_pre_referral_docs,f"{self.save_dir}/processed_pre_referral_instances_long_docs.csv")

        # self.write_filtered_to_file(self.processed_post_referral_data,f"{self.save_dir}/processed_post_referral_long.csv")
        # concatenated and grouped by brc_id data
        self.write_filtered_to_file(self.grouped_processed_pre_referral_data,f"{self.save_dir}/processed_pre_referral_grouped_subject.csv")
        # self.write_filtered_to_file(self.grouped_processed_post_referral_data,f"{self.save_dir}/processed_post_referral_concatenated.csv")

        #save the instance concatenated to file
        self.write_filtered_to_file(self.instance_concat_pre_referral_data,f"{self.save_dir}/processed_pre_referral_instances_concat.csv")
        self.write_filtered_to_file(self.instance_concat_pre_referral_docs,f"{self.save_dir}/processed_pre_referral_instances_concat_docs.csv")


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--data_dir",
                        default = "F:/OxfordTempProjects/PatientTriageNLP/raw_data/",
                        type=str,
                        help = "The data path to the directory containing the notes and referral data files")
    parser.add_argument("--patient_cohort_path",
                        default = "1_patient_cohort.xlsx",
                        type=str,
                        help = "The data path to the file containing patient cohort data file")
    parser.add_argument("--pre_referral_data_path",
                        default = "2_notes_pre_referral.csv",
                        type=str,
                        help = "The data path to the file containing the pre referral notes data files")
    parser.add_argument("--post_referral_data_path",
                        default = "3_notes_0_days_to_28_days_post_referral.csv",
                        type=str,
                        help = "The data path to the directory containing post referral notes data files")
    parser.add_argument("--pre14_post28_attachment_path",
                        default = "4_attachments_14_days_prior_to_28_days_post_referral.csv",
                        type=str,
                        help = "The data path to the directory containing post referral notes data files")
    parser.add_argument("--save_dir",
                        default = "F:/OxfordTempProjects/PatientTriageNLP/processed_data/",
                        type=str,
                        help = "The directory to save processed and cleaned data files")
    parser.add_argument("--admin_language",
                        default = [ "FINAL REPORT",
                                    "Date/Time",
                                    "Phone",
                                    "Date of Birth",
                                    "DoB",
                                    "Completed by",
                                    "Dictated by",
                                    "name of assessor:",
                                    "assessed by",
                                    "private and confidential", '\t'],
                        type=list,
                        help = "User defined list of strings to replace during cleaning using regular expression")
    parser.add_argument("--sample",
                        action = "store_true",
                        help = "Whether or not to process a sub sample of brc_ids")
    parser.add_argument("--sample_size",
                        default = 10,
                        type=int,
                        help = "The sample size to use when subsetting")

    parser.add_argument("--lookback_window",
                        default = 0,
                        type=int,
                        help = "The nnumber of days to lookback for instance documents")

    parser.add_argument("--chronological",
                        action = "store_true",
                        help = "Whether or not to keep documents in chronological order before concatenation. True means yes. False will sort with most recent first.")


    args = parser.parse_args()

    print(f"arguments are: {args}")

    if args.chronological:
        logger.warning("Chronological set to true. Concatenated document data will be kept in form: earliest - latest")
        chronological = True
    else:
        chronological = False

    if args.lookback_window >0:
        logger.warning(f"clinical documents related to an instance will be restricted to a lookback window of: {args.lookback_window} days! ")

    clinical_textdata = TextData(data_dir = args.data_dir,patient_cohort_path=args.patient_cohort_path,
                                 pre_referral_data_path=args.pre_referral_data_path, post_referral_data_path=args.post_referral_data_path,
                                 pre14_prior28_attachments_path=args.pre14_post28_attachment_path,
                                 save_dir = args.save_dir, sample = args.sample, sample_size = args.sample_size, admin_language=args.admin_language)
    clinical_textdata.clean_save_data(lookback_window=args.lookback_window, chronological=chronological)
if __name__ == "__main__":
    main()
