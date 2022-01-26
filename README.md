# Repo to contain code relating to prompt based models for Mimic-III/Biomedical tasks

The plan as it stands is to conduct a review of state of the art text classification NLP models such as BERT and T5 based models to a benchmark clinical dataset in MIMIC-III and later extend these to NHS datasets/Chronos.

We will be exploring the use of prompt-based learning as the new approach to tackling these downstream tasks. We will explore the zero shot, few shot and full fine tuning scenarios.


## Mimic-III ICD9 diagnosis code classification 

This is a multi-class classification problem where discharge summaries from ICU are used to classify the primary diagnosis code. Similar to the task here: https://github.com/simonlevine/clinical-longformer.

We are going to go with this task of classifying the top 50 diagnoses are a start, but will also develop a novel "triage" oriented task with the same data by grouping the ICD9 codes into clinically similar disease groups i.e. treatment pathways. 

### Data directory setup

Data cannot be stored here, so access to the raw mimic-iii data will be required.

![image](https://user-images.githubusercontent.com/49034323/151138574-05e97f18-b1c1-4a8f-808b-b8ebd0265148.png)


### Formatting for icd9 top 50 classification

... UPDATE


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



