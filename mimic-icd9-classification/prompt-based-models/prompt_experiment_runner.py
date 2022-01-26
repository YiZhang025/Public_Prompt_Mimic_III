from tqdm import tqdm
from openprompt.data_utils import PROCESSORS
import torch
from openprompt.data_utils.utils import InputExample
import argparse
import numpy as np

from openprompt import PromptDataLoader
from openprompt.prompts import ManualVerbalizer, ManualTemplate, SoftVerbalizer

from openprompt.prompts import SoftTemplate
from openprompt import PromptForClassification

from utils import Mimic_ICD9_Processor as MimicProcessor
import time
import os
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

import torchmetrics.functional.classification as metrics
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix



'''
Script to run different setups of prompt learning.

Right now this is primarily set up for the mimic_top50_icd9 task, although it is quite flexible to other datasets. Any datasets need a corresponding processor class in utils.


example usage. python prompt_experiment_runner.py --model bert --model_name_or_path bert-base-uncased --num_epochs 10 --tune_plm


'''


 # create a args parser with required arguments.
parser = argparse.ArgumentParser("")
parser.add_argument("--shot", type=int, default=-1)
parser.add_argument("--seed", type=int, default=144)
parser.add_argument("--plm_eval_mode", action="store_true", help="whether to turn off the dropout in the freezed model. Set to true to turn off.")
parser.add_argument("--tune_plm", action="store_true")
parser.add_argument("--model", type=str, default='t5', help="The plm to use e.g. t5-base, roberta-large, bert-base, emilyalsentzer/Bio_ClinicalBERT")
parser.add_argument("--model_name_or_path", default='t5-base')
parser.add_argument("--project_root", default="/home/niallt/NLP_DPhil/DPhil_projects/mimic-icd9-classification/prompt-based-models", help="The project root in the file system, i.e. the absolute path of OpenPrompt")
parser.add_argument("--template_id", type=int, default = 2)
parser.add_argument("--verbalizer_id", type=int, default = 0)
parser.add_argument("--template_type", type=str, default ="manual")
parser.add_argument("--verbalizer_type", type=str, default ="soft")
parser.add_argument("--data_dir", type=str, default="/home/niallt/NLP_DPhil/DPhil_projects/mimic-icd9-classification/data/intermediary-data/top_50_icd9") # sometimes, huggingface datasets can not be automatically downloaded due to network issue, please refer to 0_basic.py line 15 for solutions. 
parser.add_argument("--dataset",type=str, default = "icd9_50")
parser.add_argument("--result_file", type=str, default="./mimic_icd9_top50/st_results/results.txt")
parser.add_argument("--scripts_path", type=str, default="./scripts/mimic_icd9_top50/")
parser.add_argument("--class_labels_file", type=str, default="./scripts/mimic_icd9_top50/labels.txt")
parser.add_argument("--max_steps", default=15000, type=int)
parser.add_argument("--prompt_lr", type=float, default=0.3)
parser.add_argument("--warmup_step_prompt", type=int, default=100)
parser.add_argument("--num_epochs", type=int, default=5)
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--init_from_vocab", action="store_true")
parser.add_argument("--eval_every_steps", type=int, default=100)
parser.add_argument("--soft_token_num", type=int, default=20)
parser.add_argument("--optimizer", type=str, default="Adafactor")

# instatiate args and set to variable
args = parser.parse_args()

# write arguments to a txt file to go with the model checkpoint and logs
content_write = "="*20+"\n"
content_write += f"dataset {args.dataset}\t"
content_write += f"temp {args.template_id}\t"
content_write += f"verb {args.verbalizer_id}\t"
content_write += f"model {args.model}\t"
content_write += f"seed {args.seed}\t"
content_write += f"shot {args.shot}\t"
content_write += f"plm_eval_mode {args.plm_eval_mode}\t"
content_write += f"init_from_vocab {args.init_from_vocab}\t"
content_write += f"eval_every_steps {args.eval_every_steps}\t"
content_write += f"prompt_lr {args.prompt_lr}\t"
content_write += f"optimizer {args.optimizer}\t"
content_write += f"warmup_step_prompt {args.warmup_step_prompt}\t"
content_write += f"soft_token_num {args.soft_token_num}\t"
content_write += "\n"

print(content_write)

import random

from openprompt.utils.reproduciblity import set_seed
set_seed(args.seed)

from openprompt.plms.seq2seq import T5TokenizerWrapper, T5LMTokenizerWrapper
from transformers import T5Config, T5Tokenizer, T5ForConditionalGeneration
from openprompt.data_utils.data_sampler import FewShotSampler
from openprompt.plms import load_plm


# set up some variables to add to checkpoint and logs filenames
time_now = str(datetime.now().strftime("%d-%m-%Y--%H-%M"))
version = f"version_{time_now}"

plm, tokenizer, model_config, WrapperClass = load_plm(args.model, args.model_name_or_path)

# edit based on whether or not plm was frozen during training
if args.tune_plm == True:
    print("Unfreezing the plm - will be updated during training")
    freeze_plm = False
    # set checkpoint, logs and results save_dirs
    ckpt_dir = f"{args.project_root}/checkpoints/{args.model_name_or_path}_temp{args.template_type}{args.template_id}_verb{args.verbalizer_type}{args.verbalizer_id}/{version}"
    logs_dir = f"{args.project_root}/logs/{args.model_name_or_path}_temp{args.template_type}{args.template_id}_verb{args.verbalizer_type}{args.verbalizer_id}/{version}"
    results_dir = f"{args.project_root}/results/{args.model_name_or_path}_temp{args.template_type}{args.template_id}_verb{args.verbalizer_type}{args.verbalizer_id}/{version}"
else:
    print("Freezing the plm")
    freeze_plm = True
    # set checkpoint, logs and results save_dirs
    ckpt_dir = f"{args.project_root}/checkpoints/frozen_plm/{args.model_name_or_path}_temp{args.template_type}{args.template_id}_verb{args.verbalizer_type}{args.verbalizer_id}/{version}"
    logs_dir = f"{args.project_root}/logs/frozen_plm/{args.model_name_or_path}_temp{args.template_type}{args.template_id}_verb{args.verbalizer_type}{args.verbalizer_id}/{version}"
    results_dir = f"{args.project_root}/results/frozen_plm/{args.model_name_or_path}_temp{args.template_type}{args.template_id}_verb{args.verbalizer_type}{args.verbalizer_id}/{version}"

# check if the checkpoint and results dir exists  

if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# set up tensorboard logger
writer = SummaryWriter(logs_dir)

dataset = {}

# Below are multiple dataset examples, although right now just mimic ic9-top50. 
if args.dataset == "icd9_50":
    Processor = MimicProcessor
    # get different splits
    dataset['train'] = Processor().get_examples(data_dir = args.data_dir, mode = "train")
    dataset['validation'] = Processor().get_examples(data_dir = args.data_dir, mode = "valid")
    dataset['test'] = Processor().get_examples(data_dir = args.data_dir, mode = "test")
    # the below class labels should align with the label encoder fitted to training data
    # you will need to generate this class label text file first using the mimic processor with generate_class_labels flag to set true
    # e.g. Processor().get_examples(data_dir = args.data_dir, mode = "train", generate_class_labels = True)[:10000]
    class_labels =Processor().load_class_labels(file_path = args.class_labels_file)
    print(f"number of classes: {len(class_labels)}")
    scriptsbase = args.scripts_path
    scriptformat = "txt"
    max_seq_l = 480 # this should be specified according to the running GPU's capacity 
    if args.tune_plm: # tune the entire plm will use more gpu-memories, thus we should use a smaller batch_size.
        batchsize_t = args.batch_size 
        batchsize_e = args.batch_size
        gradient_accumulation_steps = 4
        model_parallelize = False # if multiple gpus are available, one can use model_parallelize
    else:
        batchsize_t = args.batch_size
        batchsize_e = args.batch_size
        gradient_accumulation_steps = 4
        model_parallelize = False
else:

    #TODO implement mimic readmission
    raise NotImplementedError


# Now define the template and verbalizer. 
# Note that soft template can be combined with hard template, by loading the hard template from file. 
# For example, the template in soft_template.txt is {}
# The choice_id 1 is the hard template 

# decide which template and verbalizer to use
if args.template_type == "manual":
    print(f"manual template selected, with id :{args.template_id}")
    mytemplate = ManualTemplate(tokenizer=tokenizer).from_file(f"{scriptsbase}/manual_template.txt", choice=args.template_id)

elif args.template_type == "soft":
    print(f"soft template selected, with id :{args.template_id}")
    mytemplate = SoftTemplate(model=plm, tokenizer=tokenizer, num_tokens=args.soft_token_num, initialize_from_vocab=args.init_from_vocab).from_file(f"{scriptsbase}/soft_template.txt", choice=args.template_id)

# now set verbalizer
if args.verbalizer_type == "manual":
    print(f"manual verbalizer selected, with id :{args.verbalizer_id}")
    myverbalizer = ManualVerbalizer(tokenizer, classes=class_labels).from_file(f"{scriptsbase}/manual_verbalizer.{scriptformat}", choice=args.verbalizer_id)

elif args.verbalizer_type == "soft":
    print(f"soft verbalizer selected!")
    myverbalizer = SoftVerbalizer(tokenizer, plm, num_classes=len(class_labels))

wrapped_example = mytemplate.wrap_one_example(dataset['train'][0]) 
print(wrapped_example)


use_cuda = True
#TODO rename tune_plm to freeze_plm... 


print(f"tune_plm value: {args.tune_plm}")
prompt_model = PromptForClassification(plm=plm,template=mytemplate, verbalizer=myverbalizer, freeze_plm=freeze_plm, plm_eval_mode=args.plm_eval_mode)
if use_cuda:
    prompt_model=  prompt_model.cuda()

if model_parallelize:
    prompt_model.parallelize()


train_dataloader = PromptDataLoader(dataset=dataset["train"], template=mytemplate, tokenizer=tokenizer, 
    tokenizer_wrapper_class=WrapperClass, max_seq_length=max_seq_l, decoder_max_length=3, 
    batch_size=batchsize_t,shuffle=True, teacher_forcing=False, predict_eos_token=False,
    truncate_method="tail")

validation_dataloader = PromptDataLoader(dataset=dataset["validation"], template=mytemplate, tokenizer=tokenizer, 
    tokenizer_wrapper_class=WrapperClass, max_seq_length=max_seq_l, decoder_max_length=3, 
    batch_size=batchsize_e,shuffle=False, teacher_forcing=False, predict_eos_token=False,
    truncate_method="tail")

# zero-shot test
test_dataloader = PromptDataLoader(dataset=dataset["test"], template=mytemplate, tokenizer=tokenizer, 
    tokenizer_wrapper_class=WrapperClass, max_seq_length=max_seq_l, decoder_max_length=3, 
    batch_size=batchsize_e,shuffle=False, teacher_forcing=False, predict_eos_token=False,
    truncate_method="tail")

print("truncate rate: {}".format(test_dataloader.tokenizer_wrapper.truncate_rate), flush=True)

from transformers import  AdamW, get_linear_schedule_with_warmup,get_constant_schedule_with_warmup  # use AdamW is a standard practice for transformer 
from transformers.optimization import Adafactor, AdafactorSchedule  # use Adafactor is the default setting for T5
loss_func = torch.nn.CrossEntropyLoss()

tot_step = args.max_steps

if args.tune_plm:
    
    print("We will be tuning the PLM!") # normally we freeze the model when using soft_template. However, we keep the option to tune plm
    no_decay = ['bias', 'LayerNorm.weight'] # it's always good practice to set no decay to biase and LayerNorm parameters
    optimizer_grouped_parameters_plm = [
        {'params': [p for n, p in prompt_model.plm.named_parameters() if (not any(nd in n for nd in no_decay))], 'weight_decay': 0.01},
        {'params': [p for n, p in prompt_model.plm.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer_plm = AdamW(optimizer_grouped_parameters_plm, lr=3e-5)
    scheduler_plm = get_linear_schedule_with_warmup(
        optimizer_plm, 
        num_warmup_steps=100, num_training_steps=tot_step)
else:
    print("We will not be tunning the plm - i.e. the PLM layers are frozen during training")
    optimizer_plm = None
    scheduler_plm = None

# if using soft template
if args.template_type == "soft":
    print("Soft template used - will be fine tuning the prompt embeddings!")
    optimizer_grouped_parameters_template = [{'params': [p for name, p in prompt_model.template.named_parameters() if 'raw_embedding' not in name]}] # note that you have to remove the raw_embedding manually from the optimization
    if args.optimizer.lower() == "adafactor":
        optimizer_template = Adafactor(optimizer_grouped_parameters_template,  
                                lr=args.prompt_lr,
                                relative_step=False,
                                scale_parameter=False,
                                warmup_init=False)  # when lr is 0.3, it is the same as the configuration of https://arxiv.org/abs/2104.08691
        scheduler_template = get_constant_schedule_with_warmup(optimizer_template, num_warmup_steps=args.warmup_step_prompt) # when num_warmup_steps is 0, it is the same as the configuration of https://arxiv.org/abs/2104.08691
    elif args.optimizer.lower() == "adamw":
        optimizer_template = AdamW(optimizer_grouped_parameters_template, lr=args.prompt_lr) # usually lr = 0.5
        scheduler_template = get_linear_schedule_with_warmup(
                        optimizer_template, 
                        num_warmup_steps=args.warmup_step_prompt, num_training_steps=tot_step) # usually num_warmup_steps is 500

elif args.template_type == "manual":
    optimizer_template = None
    scheduler_template = None


if args.verbalizer_type == "soft":
    print("Soft verbalizer used - will be fine tuning the verbalizer/answer embeddings!")
    optimizer_grouped_parameters_verb = [
    {'params': prompt_model.verbalizer.group_parameters_1, "lr":3e-5},
    {'params': prompt_model.verbalizer.group_parameters_2, "lr":3e-4},
    
    ]
    optimizer_verb= AdamW(optimizer_grouped_parameters_verb)
    scheduler_verb = get_linear_schedule_with_warmup(
                        optimizer_verb, 
                        num_warmup_steps=args.warmup_step_prompt, num_training_steps=tot_step) # usually num_warmup_steps is 500

elif args.verbalizer_type == "manual":
    optimizer_verb = None
    scheduler_verb = None


def train(prompt_model, train_dataloader, num_epochs, mode = "train", ckpt_dir = ckpt_dir):

    # set model to train 
    prompt_model.train()

    # set up some counters
    actual_step = 0
    glb_step = 0

    # some validation metrics to monitor
    best_val_acc = 0
    best_val_f1 = 0
    best_val_prec = 0    
    best_val_recall = 0

    # this will be set to true when max steps are reached
    leave_training = False

    for epoch in tqdm(range(num_epochs)):
        print(f"On epoch: {epoch}")
        tot_loss = 0 
        epoch_loss = 0

        for step, inputs in enumerate(train_dataloader):
            if use_cuda:
                inputs = inputs.cuda()
            logits = prompt_model(inputs)
            labels = inputs['label']
            # print(f"labels : {labels}")
            loss = loss_func(logits, labels)
            loss.backward()
            tot_loss += loss.item()

            actual_step+=1
            # clip gradients based on gradient accumulation steps
            if actual_step % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(prompt_model.parameters(), 1.0)
                glb_step += 1

            # log loss to tensorboard  every 50 steps  
            if step %50 ==49:
               
                aveloss = tot_loss/(step+1)
                # write to tensorboard
                writer.add_scalar("train/batch_loss", aveloss, glb_step)                
            
                print("Epoch {}, average loss: {}".format(epoch, tot_loss/(step+1)), flush=True)   

            # backprop the loss and update optimizers and then schedulers too
            # plm
            if optimizer_plm is not None:
                optimizer_plm.step()
                optimizer_plm.zero_grad()
            if scheduler_plm is not None:
                scheduler_plm.step()
            # template
            if optimizer_template is not None:
                optimizer_template.step()
                optimizer_template.zero_grad()
            if scheduler_template is not None:
                scheduler_template.step()
            # verbalizer
            if optimizer_verb is not None:
                optimizer_verb.step()
                optimizer_verb.zero_grad()
            if scheduler_verb is not None:
                scheduler_verb.step()

            # check if we are over max steps
            if glb_step > args.max_steps:
                leave_training = True
                break

        
        # get epoch loss and write to tensorboard

        epoch_loss = tot_loss/len(train_dataloader)
        writer.add_scalar("train/epoch_loss", epoch_loss, epoch)

        # run a run through validation set to get some metrics        
        val_acc, val_prec, val_recall, val_f1 = evaluate(prompt_model, validation_dataloader)

        writer.add_scalar("valid/accuracy", val_acc, epoch)
        writer.add_scalar("valid/precision", val_prec, epoch)
        writer.add_scalar("valid/recall", val_recall, epoch)
        writer.add_scalar("valid/f1", val_f1, epoch)

        # save checkpoint if validation accuracy improved
        if val_acc >= best_val_acc:
            print("Accuracy improved! Saving checkpoint!")
            torch.save(prompt_model.state_dict(),f"{ckpt_dir}/checkpoint.ckpt")
            best_val_acc = val_acc


        if glb_step > args.max_steps:
            leave_training = True
            break
    
        if leave_training:
            print("Leaving training as max steps have been met!")
            break 

   

# ## evaluate

# %%

def evaluate(prompt_model, dataloader, mode = "validation"):
    prompt_model.eval()

    allpreds = []
    alllabels = []
    with torch.no_grad():
        for step, inputs in enumerate(dataloader):
            if use_cuda:
                inputs = inputs.cuda()
            logits = prompt_model(inputs)
            labels = inputs['label']
            alllabels.extend(labels.cpu().tolist())
            allpreds.extend(torch.argmax(logits, dim=-1).cpu().tolist())

    acc = sum([int(i==j) for i,j in zip(allpreds, alllabels)])/len(allpreds)
    print(f"accuracy using manual method: {acc}")

    # below is torch metrics but needs to still be tensors
    # f1 = metrics.f1(allpreds,alllabels, average = 'weighted', num_classes = len(class_labels))
    # prec =metrics.precision(allpreds,alllabels, average = 'weighted', num_classes =len(class_labels))
    # recall = metrics.recall(allpreds,alllabels, average = 'weighted', num_classes = len(class_labels))
    # acc = metrics.accuracy(allpreds,alllabels, average = 'weighted', num_classes = len(class_labels))
    
    # get sklearn based metrics
    f1 = f1_score(alllabels, allpreds, average = 'weighted')
    prec = precision_score(alllabels, allpreds, average = 'weighted')
    recall = recall_score(alllabels, allpreds, average = 'weighted')   
    
    return acc, prec, recall, f1


# run training

train(prompt_model, train_dataloader, args.num_epochs, ckpt_dir)

# write the contents to file

print(content_write)

with open(f"{results_dir}/results.txt", "a") as fout:
    fout.write(content_write)

writer.flush()

# allpreds = []
# alllabels = []
# with torch.no_grad():
#     for step, inputs in enumerate(test_dataloader):
#         if use_cuda:
#             inputs = inputs.cuda()
#         logits = prompt_model(inputs)
#         labels = inputs['label']
#         alllabels.extend(labels.cpu().tolist())
#         allpreds.extend(torch.argmax(logits, dim=-1).cpu().tolist())
# acc = sum([int(i==j) for i,j in zip(allpreds, alllabels)])/len(allpreds)
# print("test:", acc)  
# writer.add_scalar("test/accuracy", acc, epoch)

##############end of code from soft_verbalizer script ##################################

#  # TODO - debug below - it does not attain the performance it should compared to the loop above
# # TODO - rename below to optimizer_plm and scheduler_plm


# from transformers import  AdamW, get_linear_schedule_with_warmup,get_constant_schedule_with_warmup  # use AdamW is a standard practice for transformer 
# from transformers.optimization import Adafactor, AdafactorSchedule  # use Adafactor is the default setting for T5
# loss_func = torch.nn.CrossEntropyLoss()

# tot_step = args.max_steps
# if args.tune_plm:
    
#     print("We will be tuning the PLM!") # normally we freeze the model when using soft_template. However, we keep the option to tune plm
#     no_decay = ['bias', 'LayerNorm.weight'] # it's always good practice to set no decay to biase and LayerNorm parameters
#     optimizer_grouped_parameters_plm = [
#         {'params': [p for n, p in prompt_model.plm.named_parameters() if (not any(nd in n for nd in no_decay))], 'weight_decay': 0.01},
#         {'params': [p for n, p in prompt_model.plm.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
#     ]
#     optimizer_plm = AdamW(optimizer_grouped_parameters_plm, lr=3e-5)
#     scheduler_plm = get_linear_schedule_with_warmup(
#         optimizer_plm, 
#         num_warmup_steps=100, num_training_steps=tot_step)
# else:
#     print("We will not be tunning the plm - i.e. the PLM layers are frozen during training")
#     optimizer_plm = None
#     scheduler_plm = None

# # if using soft template
# if args.template_type == "soft":
#     optimizer_grouped_parameters_template = [{'params': [p for name, p in prompt_model.template.named_parameters() if 'raw_embedding' not in name]}] # note that you have to remove the raw_embedding manually from the optimization
#     if args.optimizer.lower() == "adafactor":
#         optimizer_template = Adafactor(optimizer_grouped_parameters_template,  
#                                 lr=args.prompt_lr,
#                                 relative_step=False,
#                                 scale_parameter=False,
#                                 warmup_init=False)  # when lr is 0.3, it is the same as the configuration of https://arxiv.org/abs/2104.08691
#         scheduler_template = get_constant_schedule_with_warmup(optimizer_template, num_warmup_steps=args.warmup_step_prompt) # when num_warmup_steps is 0, it is the same as the configuration of https://arxiv.org/abs/2104.08691
#     elif args.optimizer.lower() == "adamw":
#         optimizer_template = AdamW(optimizer_grouped_parameters_template, lr=args.prompt_lr) # usually lr = 0.5
#         scheduler_template = get_linear_schedule_with_warmup(
#                         optimizer_template, 
#                         num_warmup_steps=args.warmup_step_prompt, num_training_steps=tot_step) # usually num_warmup_steps is 500

# elif args.template_type == "manual":
#     optimizer_template = None
#     scheduler_template = None


# if args.verbalizer_type == "soft":
#     optimizer_grouped_parameters_verb = [
#     {'params': prompt_model.verbalizer.group_parameters_1, "lr":3e-5},
#     {'params': prompt_model.verbalizer.group_parameters_2, "lr":3e-4},
    
#     ]
#     optimizer_verb= AdamW(optimizer_grouped_parameters_verb)
#     scheduler_verb = get_linear_schedule_with_warmup(
#                         optimizer_verb, 
#                         num_warmup_steps=args.warmup_step_prompt, num_training_steps=tot_step) # usually num_warmup_steps is 500

# elif args.verbalizer_type == "manual":
#     optimizer_verb = None
#     scheduler_verb = None


# # training loop
# tot_loss = 0 
# log_loss = 0
# best_val_acc = 0
# glb_step = 0
# actual_step = 0
# leave_training = False

# acc_traces = []
# tot_train_time = 0
# pbar_update_freq = 10
# prompt_model.train()

# pbar = tqdm(total=tot_step, desc="Train")
# for epoch in range(100000):
#     print(f"Begin epoch {epoch}")
#     for step, inputs in enumerate(train_dataloader):
#         if use_cuda:
#             inputs = inputs.cuda()
#         tot_train_time -= time.time()
#         logits = prompt_model(inputs)
#         labels = inputs['label']
#         loss = loss_func(logits, labels)
#         loss.backward()
#         tot_loss += loss.item()
#         actual_step += 1

#         if actual_step % gradient_accumulation_steps == 0:
#             torch.nn.utils.clip_grad_norm_(prompt_model.parameters(), 1.0)
#             glb_step += 1
#             if glb_step % pbar_update_freq == 0:
#                 aveloss = (tot_loss - log_loss)/pbar_update_freq
#                 pbar.update(10)
#                 pbar.set_postfix({'loss': aveloss})
#                 log_loss = tot_loss

#                 # write to tensorboard
#                 writer.add_scalar("train/loss", aveloss, glb_step)

#         # now use the loss to update the optimizers
#         # plm
#         if optimizer_plm is not None:
#             optimizer_plm.step()
#             optimizer_plm.zero_grad()
#         if scheduler_plm is not None:
#             scheduler_plm.step()
#         # template
#         if optimizer_template is not None:
#             optimizer_template.step()
#             optimizer_template.zero_grad()
#         if scheduler_template is not None:
#             scheduler_template.step()
#         # verbalizer
#         if optimizer_verb is not None:
#             optimizer_verb.step()
#             optimizer_verb.zero_grad()
#         if scheduler_verb is not None:
#             scheduler_verb.step()



#         tot_train_time += time.time()

#         if actual_step % gradient_accumulation_steps == 0 and glb_step >0 and glb_step % args.eval_every_steps == 0:
#             val_acc = evaluate(prompt_model, validation_dataloader, desc="Valid")
#             if val_acc >= best_val_acc:
#                 torch.save(prompt_model.state_dict(),f"{ckpt_dir}.ckpt")
#                 best_val_acc = val_acc
            
#             acc_traces.append(val_acc)
#             print("Glb_step {}, val_acc {}, average time {}".format(glb_step, val_acc, tot_train_time/actual_step ), flush=True)
#             prompt_model.train()

#         if glb_step > args.max_steps:
#             leave_training = True
#             break
    
#     if leave_training:
#         break  
    
    
# # # super_glue test split can not be evaluated without submitting the results to their website. So we skip it here and keep them as comments.
# #
# # prompt_model.load_state_dict(torch.load(f"{args.project_root}/ckpts/{this_run_unicode}.ckpt"))
# # prompt_model = prompt_model.cuda()
# # test_acc = evaluate(prompt_model, test_dataloader, desc="Test")
# # test_acc = evaluate(prompt_model, test_dataloader, desc="Test")

# # a simple measure for the convergence speed.
# thres99 = 0.99*best_val_acc
# thres98 = 0.98*best_val_acc
# thres100 = best_val_acc
# step100=step98=step99=args.max_steps
# for val_time, acc in enumerate(acc_traces):
#     if acc>=thres98:
#         step98 = min(val_time*args.eval_every_steps, step98)
#         if acc>=thres99:
#             step99 = min(val_time*args.eval_every_steps, step99)
#             if acc>=thres100:
#                 step100 = min(val_time*args.eval_every_steps, step100)


# content_write += f"BestValAcc:{best_val_acc}\tEndValAcc:{acc_traces[-1]}\tcritical_steps:{[step98,step99,step100]}\n"
# content_write += "\n"

# print(content_write)

# with open(f"{args.result_file}", "a") as fout:
#     fout.write(content_write)

# import os
# os.remove(f"../ckpts/{this_run_unicode}.ckpt")