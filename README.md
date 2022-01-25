# Repo to contain code relating to prompt based models for Mimic-III/Biomedical tasks

The plan as it stands is to conduct a review of state of the art text classification NLP models such as BERT and GPT based models to a benchmark clinical dataset in MIMIC-III
and later extend these to NHS datasets/Chronos.

For more detailed plans regarding this codebase - see issues, which are much easier to update and track than readme.


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



