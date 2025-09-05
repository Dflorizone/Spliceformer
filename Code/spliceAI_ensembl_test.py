# This notebook was used to test SpliceAI model on ENSEMBL data
import numpy as np
import sys
import time
import h5py
from tqdm import tqdm

import numpy as np
import re
from math import ceil
from sklearn.metrics import average_precision_score
from torch.utils.data import Dataset
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import pickle
#import pickle5 as pickle

from sklearn.model_selection import train_test_split

from scipy.sparse import load_npz
from glob import glob

from transformers import get_constant_schedule_with_warmup
from sklearn.metrics import precision_score,recall_score,accuracy_score
import copy

from src.train import trainModel
#from src.dataloader import getData,spliceDataset,h5pyDataset,collate_fn
from src.dataloader import getData,spliceDataset,h5pyDataset,getDataPointList,getDataPointListFull,DataPointFull
from src.weight_init import keras_init
from src.losses import categorical_crossentropy_2d
from src.model import SpliceFormer, SpliceAI_10K
from src.evaluation_metrics import print_topl_statistics,cross_entropy_2d
from src.gpu_metrics import run_bootstrap, calculate_ap, calculate_topk
import os


data_dir = os.path.join(os.environ['SLURM_TMPDIR'], 'Data')
out_path = os.path.join(os.environ['SLURM_TMPDIR'], 'Data', 'spliceai_10k_test_ensembl_predictions_270825.csv.gz')
# if not os.path.exists(out_path):
#     with open(out_path, 'wt') as f:
#         f.write("Y_true_acceptor,Y_pred_acceptor,Y_true_donor,Y_pred_donor\n")
buffer = []
flush = 500
first_write = True

CL_max=10000
SL=5000
BATCH_SIZE = 16
NUM_ACCUMULATION_STEPS=1
setType = 'test'
annotation, transcriptToLabel, seqData = getData(data_dir, setType)



temp = 1
n_models = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_m = SpliceAI_10K(CL_max)
model_m.apply(keras_init)
model_m = model_m.to(device)

if torch.cuda.device_count() > 1:
    model_m = nn.DataParallel(model_m)
model_m = nn.DataParallel(model_m)
output_class_labels = ['Null', 'Acceptor', 'Donor']

#for output_class in [1,2]:
models = [copy.deepcopy(model_m) for i in range(n_models)]
[model.load_state_dict(torch.load('../Results/PyTorch_Models/spliceai_encoder_10k_191022_{}'.format(i))) for i,model in enumerate(models)]
#nr = [0,2,3]
#[model.load_state_dict(torch.load('../Results/PyTorch_Models/transformer_encoder_40k_201221_{}'.format(nr[i]))) for i,model in enumerate(models)]
#chunkSize = num_idx/10
for model in models:
    model.eval()

Y_true_acceptor, Y_pred_acceptor = [],[]
Y_true_donor, Y_pred_donor = [],[]
test_dataset = spliceDataset(getDataPointListFull(annotation,transcriptToLabel,SL,CL_max,shift=SL))
test_dataset.seqData = seqData
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=16, pin_memory=True)


#targets_list = []
#outputs_list = []
ce_2d = []
with torch.no_grad():
    for (batch_features ,targets) in tqdm(test_loader):
        batch_features = batch_features.type(torch.FloatTensor).to(device)
        targets = targets.to(device)[:,:,CL_max//2:-CL_max//2]
        outputs = ([models[i](batch_features).detach() for i in range(n_models)])
        #outputs = (outputs[0]+outputs[1]+outputs[2]+outputs[3]+outputs[4])/n_models
        outputs = torch.stack(outputs)
        outputs = torch.mean(outputs,dim=0)
        #outputs = odds_gmean(outputs)
        #targets_list.extend(targets.unsqueeze(0))
        #outputs_list.extend(outputs.unsqueeze(0))

        targets = torch.transpose(targets,1,2).cpu().numpy()
        outputs = torch.transpose(outputs,1,2).cpu().numpy()
        ce_2d.append(cross_entropy_2d(targets,outputs))

        is_expr = (targets.sum(axis=(1,2)) >= 1)
        # Y_true_acceptor.extend(targets[is_expr, :, 1].flatten())
        # Y_true_donor.extend(targets[is_expr, :, 2].flatten())
        # Y_pred_acceptor.extend(outputs[is_expr, :, 1].flatten())
        # Y_pred_donor.extend(outputs[is_expr, :, 2].flatten())

        Y_true_acceptor = targets[is_expr, :, 1].flatten()
        Y_true_donor = targets[is_expr, :, 2].flatten()
        Y_pred_acceptor = outputs[is_expr, :, 1].flatten()
        Y_pred_donor = outputs[is_expr, :, 2].flatten()

        df = pd.DataFrame({'Y_true_acceptor':Y_true_acceptor,'Y_pred_acceptor':Y_pred_acceptor,'Y_true_donor':Y_true_donor,'Y_pred_donor':Y_pred_donor})
        buffer.append(df)
        if len(buffer) >= flush:
            pd.concat(buffer).to_csv(out_path, mode='a', index=False, header=first_write, compression='gzip')
            buffer = []
            first_write = False
if buffer:
    pd.concat(buffer).to_csv(out_path, mode='a', index=False, header=False, compression='gzip')

# Y_true_acceptor, Y_pred_acceptor,Y_true_donor, Y_pred_donor = np.array(Y_true_acceptor), np.array(Y_pred_acceptor),np.array(Y_true_donor), np.array(Y_pred_donor)
# df = pd.DataFrame({'Y_true_acceptor':Y_true_acceptor,'Y_pred_acceptor':Y_pred_acceptor,'Y_true_donor':Y_true_donor,'Y_pred_donor':Y_pred_donor})
# df_path = os.path.join(os.environ['SLURM_TMPDIR'], 'transformer_40k_test_ensembl_predictions_260625.gz')
# df.to_csv(df_path,index=False)