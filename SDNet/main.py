# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import argparse
import os
import sys
import torch
import csv
from Models.SDNetTrainer import SDNetTrainer
from Utils.Arguments import Arguments

opt = None

parser = argparse.ArgumentParser(description='SDNet')
parser.add_argument('command', help='Command: train')
parser.add_argument('conf_file', help='Path to conf file.')

cmdline_args = parser.parse_args()
command = cmdline_args.command
conf_file = cmdline_args.conf_file
conf_args = Arguments(conf_file)
opt = conf_args.readArguments()
opt['cuda'] = torch.cuda.is_available()
opt['confFile'] = conf_file
opt['datadir'] = os.path.dirname(conf_file)  # conf_file specifies where the data folder is

for key,val in cmdline_args.__dict__.items():
    if val is not None and key not in ['command', 'conf_file']:
        opt[key] = val

model = SDNetTrainer(opt)    
    
print('Select command: ' + command)
if command == 'official':
    print("Official Test Run")
    with open(opt['OFFICIAL_TEST_FILE'], 'r') as f:
        test_data = json.load(f)
        _, _, _, submission = model.official(opt['MODEL_PATH'], test_data)

        sub_path = opt['SAVE_DIR'] + opt['SUB_FILE']
        with open(sub_path, 'w') as csv_fh:
           csv_writer = csv.writer(csv_fh, delimiter=',')
           csv_writer.writerow(['Id', 'Predicted'])
           for uuid in sorted(submission):
               csv_writer.writerow([uuid,submission[uuid]])
else:
    model.train()
