# Repo to contain code relating to prompt based models for Mimic-III/Biomedical tasks

The plan as it stands is to conduct a review of state of the art text classification NLP models such as BERT and T5 based models to a benchmark clinical dataset in MIMIC-III and later extend these to NHS datasets/Chronos.

We will be exploring the use of prompt-based learning as the new approach to tackling these downstream tasks. We will explore the zero shot, few shot and full fine tuning scenarios.


## Mimic-III ICD9 diagnosis code classification 

This is a multi-class classification problem where discharge summaries from ICU are used to classify the primary diagnosis code. Similar to the task here: https://github.com/simonlevine/clinical-longformer.

We are going to go with this task of classifying the top 50 diagnoses are a start, but will also develop a novel "triage" oriented task with the same data by grouping the ICD9 codes into clinically similar disease groups i.e. treatment pathways. 

### Data directory setup

Data cannot be stored here, so access to the raw mimic-iii data will be required.

![image](https://user-images.githubusercontent.com/49034323/151138574-05e97f18-b1c1-4a8f-808b-b8ebd0265148.png)


The raw data is contained in the following: "./data/physionet.org/files/mimiciii/1.4/zipped_data/"


### Formatting for icd9 top 50 classification

To create the training/valid/test splits for the top N icd9 diagnosis code and triage classification tasks first run the following scripts in order on the raw notes data. Perform following commands from the base dir of the repo.

#### 1.)

```
python mimic-icd9-classification/preprocessing_scripts/format_notes.py
```
This will do some initial basic cleaning of the raw notes into appropriate dataframes for the different icd9 based classificaiton tasks. Compressed data will be saved alongside the original raw data as "NOTEEVENTS.FILTERED.csv.gz" by default

#### 2.)

```
python mimic-icd9-classification/preprocessing_scripts/format_data_for_training.py
```
This will organise data into appropriate dataframes for the different icd9 based classificaiton tasks - either the topNicd9 classification, or triage tasks. Train/validate/test sets will be created containing all icd9_codes/data. Data will be saved at "./mimic-icd9-classification/data/intermediary_data/note2diagnosis-icd-{train/validate/test}.csv" 

#### 3a - TopNicd9 classification 

```
python mimic-icd9-classification/preprocessing_scripts/format_mimic_topN_icd9.py
```
By default this will take the top 50 most frequent icd9 diagnosis codes as remove all other data (still contains the vast majority of the data) and place new train/validate/test splits inside the folder "/mimic-icd9-classification/data/intermediary_data/top_50_icd9/{train/validate/test}.csv"

#### 3b - Triage icd9 classification

```
python mimic-icd9-classification/preprocessing_scripts/format_mimic_icd9_triage.py
```
This is a more experimental task where we have further split icd9 diagnosis codes into groupings that reflect their disease ontology and likely department/treatment pathways.

By default this will take the top 20 most frequent icd9 diagnosis codes and group into new triage categories, as remove all other data (still contains the vast majority of the data) and place new train/validate/test splits inside the folder "/mimic-icd9-classification/data/intermediary_data/triage/{train/validate/test}.csv"


# Experiments

## Top N ICD9 classification
### standard finetuning for classification - e.g. bert/clinical-bert

Using the clinical longformer pipeline is easiest at the moment. A training script is located in the "clinical_longformer/classifier_pipeline" directory. 

To run training on the top50 icd9 classification task cd to the clinical longformer/classifier directory and run the following:

#### Clinical_Biobert with no freezing of PLM
```
python training_one_label.py --transformer_type bert --encoder_model emilysentzer/Bio_ClinicalBERT --batch_size 4 --gpus 0 --max_epochs 10 --dataset icd9_50
```

#### Clinical_Biobert with freezing of PLM
```
python training_one_label.py --transformer_type bert --encoder_model emilysentzer/Bio_ClinicalBERT --batch_size 4 --gpus 0 --max_epochs 10 --nr_frozen_epochs 10 --dataset icd9_50
```
At present this script only uses BERT based models, but can ultimately use any. There is a lot of arguments/tweaks available for this training script so you will want to investigate these within the script.

### prompt based learning
#### Clinical_BioBert with no freezing 
##### manual template and verbalizer
TODO
#### manual template and soft verbalizer
TODO
#### soft template and soft verbalizer
TODO
#### mixed template and soft verbalizer
```
python prompt_experiment_runner.py --model bert --model_name_or_path emilyalsentzer/Bio_ClinicalBERT --num_epochs 10 --template_id 0 --template_type mixed --max_steps 15000 --tune_plm
```

#### Clinical_BioBert with plm freezing 
##### manual template and verbalizer
TODO
#### manual template and soft verbalizer
TODO
#### soft template and soft verbalizer
TODO
#### mixed template and soft verbalizer
```
python prompt_experiment_runner.py --model bert --model_name_or_path emilyalsentzer/Bio_ClinicalBERT --num_epochs 10 --template_id 0 --template_type mixed --max_steps 15000 
```

## Triage ICD9 classification
### standard finetuning for classification - e.g. bert/clinical-bert

Using the clinical longformer pipeline is easiest at the moment. A training script is located in the "clinical_longformer/classifier_pipeline" directory. 

To run training on the top50 icd9 classification task cd to the clinical longformer/classifier directory and run the following:

#### Clinical_Biobert with no freezing of PLM
```
python training_onelabel.py --transformer_type bert --encoder_model emilysentzer/Bio_ClinicalBERT --batch_size 4 --gpus 0 --max_epochs 10 --dataset icd9_triage
```

#### Clinical_Biobert with freezing of PLM
```
python training_onelabel.py --transformer_type bert --encoder_model emilysentzer/Bio_ClinicalBERT --batch_size 4 --gpus 0 --max_epochs 10 --nr_frozen_epochs 10 --dataset icd9_triage
```
At present this script only uses BERT based models, but can ultimately use any. There is a lot of arguments/tweaks available for this training script so you will want to investigate these within the script.

### prompt based learning
#### Clinical_BioBert with finetuning plm
##### manual template and verbalizer
TODO
#### manual template and soft verbalizer
TODO
#### soft template and soft verbalizer
TODO
#### mixed template and soft verbalizer

#### Clinical_BioBert with plm freezing 
##### manual template and verbalizer
TODO
#### manual template and soft verbalizer
TODO
#### soft template and soft verbalizer
TODO
#### mixed template and soft verbalizer
```
python prompt_experiment_runner.py --model bert --model_name_or_path emilyalsentzer/Bio_ClinicalBERT --num_epochs 10 --template_id 0 --template_type mixed --max_steps 15000 --zero_shot --dataset icd9_triage
```

# Setup of repo on local machine

## create virtual python environment 
This will depend on OS and how python is installed. On linux can use either conda or venv. 

## with venv

```
# You only need to run this command once per-VM
sudo apt-get install python3-venv -y

# The rest of theses steps should be run every time you create
#  a new notebook (and want a virutal environment for it)

cd the/directory/your/notebook/will/be/in

# Create the virtual environment
# The '--system-site-packages' flag allows the python packages 
#  we installed by default to remain accessible in the virtual 
#  environment.  It's best to use this flag if you're doing this
#  on AI Platform Notebooks so that you keep all the pre-baked 
#  goodness
python3 -m venv myenv --system-site-packages
source myenv/bin/activate #activate the virtual env

# Register this env with jupyter lab. It’ll now show up in the
#  launcher & kernels list once you refresh the page
python -m ipykernel install --user --name=myenv

# Any python packages you pip install now will persist only in
#  this environment_
deactivate # exit the virtual env


```

## with conda

```

conda update conda

conda create -n yourenvname python=3.9 anaconda

source activate yourenvname

```

## git clone this repo

```
git clone https://github.com/NtaylorOX/Public_Prompt_Mimic_III.git 
```

## Use pip package manager
```
pip install -r requirements.txt
```


# Create new branch 

Generally we should leave master alone for developing any new code or experiments to avoid any clashes. So please follow these steps to create your own branch. Run following from bash/cmd line (wherever you ordinarily put git commands). To create a new branch as a clone of the master branch. Run following from bash/cmd line from the directory containing the git repo you have cloned (wherever you ordinarily put git commands)

```

# create new branch as clone of master
git checkout -b new_branch master

```
This should create a new branch with the name new_branch and switch automatically.



