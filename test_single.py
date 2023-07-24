from __future__ import print_function
from yaml import load, Loader
import numpy as np

import argparse
import torch
import torch.nn as nn
import pdb
import os
import pandas as pd
from utils.utils import *
from math import floor
import matplotlib.pyplot as plt
from datasets.datasets.dataset_generic import Generic_WSI_Classification_Dataset, Generic_MIL_Dataset, save_splits
import h5py
from utils.eval_utils import *

# Training settings
import argparse

# Parse the input YAML configuration file
parser = argparse.ArgumentParser(description='Configurations for WSI Testing')
parser.add_argument('config_file', type=str, help='Path to the YAML configuration file')
args = parser.parse_args()

# Load the configuration from the YAML file
with open(args.config_file, 'r') as yaml_file:
    config = load(yaml_file, Loader=Loader)

# Convert the configuration to an argparse Namespace
args = argparse.Namespace(**config)
'''
parser = argparse.ArgumentParser(description='CLAM Evaluation Script')

parser.add_argument('--data_root_dir', type=str, default='output/feat_dir',
                    help='data directory')
parser.add_argument('--results_dir', type=str, default='results',
                    help='relative path to results folder, i.e. '+
                    'the directory containing models_exp_code relative to project root (default: ./results)')
parser.add_argument('--save_exp_code', type=str, default='None',
                    help='experiment code to save eval results')
parser.add_argument('--models_exp_code', type=str, default='None',
                    help='experiment code to load trained models (directory under results_dir containing model checkpoints')
parser.add_argument('--splits_dir', type=str, default=None,
                    help='splits directory, if using custom splits other than what matches the task (default: None)')
parser.add_argument('--model_size', type=str, choices=['small', 'big'], default='small', 
                    help='size of model (default: small)')
parser.add_argument('--model_type', type=str, choices=['clam_sb', 'clam_mb', 'mil'], default='clam_sb', 
                    help='type of model (default: clam_sb)')
parser.add_argument('--drop_out', action='store_true', default=False, 
                    help='whether model uses dropout')
parser.add_argument('--k', type=int, default=10, help='number of folds (default: 10)')
parser.add_argument('--k_start', type=int, default=-1, help='start fold (default: -1, last fold)')
parser.add_argument('--k_end', type=int, default=-1, help='end fold (default: -1, first fold)')
parser.add_argument('--fold', type=int, default=-1, help='single fold to evaluate')
parser.add_argument('--micro_average', action='store_true', default=False, 
                    help='use micro_average instead of macro_avearge for multiclass AUC')
parser.add_argument('--split', type=str, choices=['train', 'val', 'test', 'all'], default='test')
parser.add_argument('--task', type=str, choices=['task_1_tumor_vs_normal',  'task_2_tumor_subtyping'])
args = parser.parse_args()
'''
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

'''
args.save_dir = os.path.join('./eval_results', 'EVAL_' + str(args.save_exp_code))
args.models_dir = os.path.join(args.results_dir, str(args.models_exp_code))
print(args.models_dir)
os.makedirs(args.save_dir, exist_ok=True)

if args.splits_dir is None:
    args.splits_dir = args.models_dir

assert os.path.isdir(args.models_dir)
assert os.path.isdir(args.splits_dir)

settings = {'task': args.task,
            'split': args.split,
            'save_dir': args.save_dir,
            'models_dir': args.models_dir,
            'model_type': args.model_type,
            'drop_out': args.drop_out,
            'model_size': args.model_size}

with open(args.save_dir + '/eval_experiment_{}.txt'.format(args.save_exp_code), 'w') as f:
    print(settings, file=f)
f.close()

print(settings)
'''
'''
source: /home/webace/Colon_Data
step_size: 256
patch_size: 256
patch: True
seg: True
stitch: True
no_auto_skip: True
save_dir: output/data_seg_patch
preset: null
patch_level: 0
process_list: null
'''
from extract_features_fp import extract_features_main
from create_patches_fp import create_patches_main
import yaml


# Rest of your imports and functions remain unchanged
class Config:
    def __init__(self, config_dict):
        for key, value in config_dict.items():
            setattr(self, key, value)


# 调用 create_features.py中的主函数
with open('create_patches_config.yaml', 'r') as file:
    args_for_CF = Config(yaml.safe_load(file))
args_for_CF.save_dir = 'test_model/output/data_seg_patch'
args_for_CF.source = 'test_model/testData'
create_patches_main(args_for_CF)

# 调用 extract_features.py 中的主函数
with open('extract_features_config.yaml', 'r') as file:
    args_for_EF = Config(yaml.safe_load(file))
args_for_EF.data_slide_dir = 'test_model/testData'
args_for_EF.data_h5_dir = 'test_model/output/data_seg_patch'
args_for_EF.csv_path = 'test_model/output/data_seg_patch/process_list_autogen.csv'
args_for_EF.feat_dir = 'test_model/output/feat_dir'

extract_features_main(args_for_EF)

if args.task == 'task_1_tumor_vs_normal':
    args.n_classes = 2
    dataset = Generic_MIL_Dataset(csv_path=args.csv,
                                  data_dir=args.data_root_dir,
                                  shuffle=False,
                                  print_info=True,
                                  label_dict={'subtype_3': 0, 'subtype_4': 1},
                                  patient_strat=False,
                                  ignore=[])

elif args.task == 'task_2_tumor_subtyping':
    args.n_classes = 3
    dataset = Generic_MIL_Dataset(csv_path=args.csv,
                                  data_dir=args.data_root_dir,
                                  shuffle=False,
                                  print_info=True,
                                  label_dict={'subtype_1': 0, 'subtype_2': 1, 'subtype_3': 2},
                                  patient_strat=False,
                                  ignore=[])

# elif args.task == 'tcga_kidney_cv':
#     args.n_classes=3
#     dataset = Generic_MIL_Dataset(csv_path = 'dataset_csv/tcga_kidney_clean.csv',
#                             data_dir= os.path.join(args.data_root_dir, 'tcga_kidney_20x_features'),
#                             shuffle = False,
#                             print_info = True,
#                             label_dict = {'TCGA-KICH':0, 'TCGA-KIRC':1, 'TCGA-KIRP':2},
#                             patient_strat= False,
#                             ignore=['TCGA-SARC'])

else:
    raise NotImplementedError
'''
if args.k_start == -1:
    start = 0
else:
    start = args.k_start
if args.k_end == -1:
    end = args.k
else:
    end = args.k_end

if args.fold == -1:
    folds = range(start, end)
else:
    folds = range(args.fold, args.fold + 1)
'''
# ckpt_paths = [os.path.join(args.models_dir, 's_{}_checkpoint.pt'.format(fold)) for fold in folds]
ckpt_paths = args.ckpt_path
# datasets_id = {'train': 0, 'val': 1, 'test': 2, 'all': -1}

if __name__ == "__main__":
    all_results = []
    all_auc = []
    all_acc = []

    model, patient_results, test_error, auc, df = eval(dataset, args, ckpt_paths)
    all_results.append(all_results)
    all_auc.append(auc)
    all_acc.append(1 - test_error)

    df.to_csv(os.path.join('{}.csv'.format(ckpt_paths)), index=False)

    # final_df = pd.DataFrame({'folds': folds, 'test_auc': all_auc, 'test_acc': all_acc})

    # final_df.to_csv(os.path.join(args.save_dir, save_name))
