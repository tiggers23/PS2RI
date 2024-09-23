import os
import torch

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ["WANDB_PROGRAM"] = "multimodal_driver.py"

DEVICE = torch.device("cuda:0")

visual_features_list=list(range(55,91)) # for mustard
acoustic_features_list=list(range(0,60)) # for mustard

ACOUSTIC_DIM = len(acoustic_features_list) # for mustard
VISUAL_DIM = len(visual_features_list) # for mustard
# VISUAL_DIM = 768 # for mustard++
# ACOUSTIC_DIM = 513 # for mustard++
HCF_DIM=4
LANGUAGE_DIM=768

VISUAL_DIM_ALL = 91 # for mustard
ACOUSTIC_DIM_ALL = 81 # for mustard
# VISUAL_DIM_ALL = 768 # for mustard++
# ACOUSTIC_DIM_ALL = 513 # for mustard++

H_MERGE_SENT = 768
DATASET_LOCATION = "./dataset/"
SEP_TOKEN_ID = 3