# Script used to test Transformer-40k on GENCODE data
import numpy as np
import h5py
from tqdm import tqdm

import numpy as np
from math import ceil
from sklearn.metrics import average_precision_score
from torch.utils.data import Dataset
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
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
from src.model import SpliceFormer
from src.evaluation_metrics import print_topl_statistics,cross_entropy_2d
from src.gpu_metrics import run_bootstrap, calculate_ap, calculate_topk
import os

SL=5000
CL_max=40000
NUM_ACCUMULATION_STEPS=1
BATCH_SIZE = 96


# data_dir = '../Data/gencode_40k_dataset_test_.h5'
data_dir = os.path.join(os.environ['SLURM_TMPDIR'], 'gencode_40k_dataset_test_.h5')
assert os.path.exists(data_dir), f"Data file not found at: {data_dir}"

h5f = h5py.File(data_dir)
num_idx = len(h5f.keys())//2

test_dataset = h5pyDataset(h5f,list(range(num_idx)))
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

temp = 1
n_models = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_m = SpliceFormer(CL_max,bn_momentum=0.01/NUM_ACCUMULATION_STEPS,depth=4,heads=4,n_transformer_blocks=2,determenistic=True)
model_m.apply(keras_init)
model_m = model_m.to(device)

if torch.cuda.device_count() > 1:
    model_m = nn.DataParallel(model_m)
model_m = nn.DataParallel(model_m)

output_class_labels = ['Null', 'Acceptor', 'Donor']

#for output_class in [1,2]:
models = [copy.deepcopy(model_m) for i in range(n_models)]
[model.load_state_dict(torch.load('../Results/PyTorch_Models/transformer_encoder_40k_171022_{}'.format(i))) for i,model in enumerate(models)]

for model in models:
    model.eval()
    
Y_true_acceptor, Y_pred_acceptor = [],[]
Y_true_donor, Y_pred_donor = [],[]
ce_2d = []
with torch.no_grad():
    for (batch_chunks,target_chunks) in tqdm(test_loader):
        batch_chunks = torch.transpose(batch_chunks[0].to(device),1,2)
        target_chunks = torch.transpose(torch.squeeze(target_chunks[0].to(device),0),1,2)
        #print(np.max(target_chunks.cpu().numpy()[:,2,:]))
        n_chunks = int(np.ceil(batch_chunks.shape[0]/BATCH_SIZE))
        batch_chunks = torch.chunk(batch_chunks, n_chunks, dim=0)
        target_chunks = torch.chunk(target_chunks, n_chunks, dim=0)
        targets_list = []
        outputs_list = []
        for j in range(len(batch_chunks)):
            batch_features = batch_chunks[j]
            targets = target_chunks[j]
            outputs = ([models[i](batch_features)[0].detach() for i in range(n_models)])
            #outputs = (outputs[0]+outputs[1]+outputs[2]+outputs[3]+outputs[4])/n_models
            outputs = torch.mean(torch.stack(outputs),dim=0)
            #outputs = odds_gmean(torch.stack(outputs))
            #outputs = (outputs[0]+outputs[1]+outputs[2])/n_models
            targets_list.extend(targets.unsqueeze(0))
            outputs_list.extend(outputs.unsqueeze(0))

        targets = torch.transpose(torch.vstack(targets_list),1,2).cpu().numpy()
        outputs = torch.transpose(torch.vstack(outputs_list),1,2).cpu().numpy()
        ce_2d.append(cross_entropy_2d(targets,outputs))

        is_expr = (targets.sum(axis=(1,2)) >= 1)
        Y_true_acceptor.extend(targets[is_expr, :, 1].flatten())
        Y_true_donor.extend(targets[is_expr, :, 2].flatten())
        Y_pred_acceptor.extend(outputs[is_expr, :, 1].flatten())
        Y_pred_donor.extend(outputs[is_expr, :, 2].flatten())


Y_true_acceptor, Y_pred_acceptor,Y_true_donor, Y_pred_donor = np.array(Y_true_acceptor), np.array(Y_pred_acceptor),np.array(Y_true_donor), np.array(Y_pred_donor)
df = pd.DataFrame({'Y_true_acceptor':Y_true_acceptor,'Y_pred_acceptor':Y_pred_acceptor,'Y_true_donor':Y_true_donor,'Y_pred_donor':Y_pred_donor})
df_path = os.path.join(os.environ['SLURM_TMPDIR'], 'transformer_40k_test_gencode_predictions_250625.gz')
df.to_csv(df_path,index=False)