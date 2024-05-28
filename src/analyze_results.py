# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np
import pickle
import sys
import torch
from sklearn import metrics

COARSE = 1
FINE = 2

fine_class_idxes = list(range(0,17))

symm_to_label_map = {'C1':0, 'C2':1,'C3':2,'C4':3,'C5':4,'C6':5,'C7':6,'C8':6,'C9':6,'C10':7,'C11':7,'C12':7,'C13':7,'C14':7,'C15':7,'C16':7,'C17':7,'D2':8,'D3':9,'D4':10,'D5':11,'D6':12,'D7':12,'D8':12,'D9':12,'D10':12,'D11':12,'D12':12,'H':13,'O':14,'T':15, 'I':16}

symm_to_coarselabel_map = {'C1':0, 'C2':1, 'D2':2, 'C3':3,'C4':3,'C5':3,'C6':3,'C7':3,'C8':3,'C9':3,'C10':3,'C11':3,'C12':3,'C13':3,'C14':3,'C15':3,'C16':3,'C17':3, 'D3':4,'D4':4,'D5':4,'D6':4,'D7':4,'D8':4,'D9':4,'D10':4,'D11':4,'D12':4,'H':5,'O':5,'T':5, 'I':5}

symm_to_joint_label_map = {'C1':0, 'C2':1,'C3':[2,17],'C4':[3,17],'C5':[4,17],'C6':[5,17],'C7':[6,17],'C8':[6,17],'C9':[6,17],'C10':[7,17], 'C11':[7,17],'C12':[7,17],'C13':[7,17],'C14':[7,17],'C15':[7,17],'C16':[7,17],'C17':[7,17],'D2':8,'D3':[9,18], 'D4':[10,18],'D5':[11,18],'D6':[12,18],'D7':[12,18],'D8':[12,18],'D9':[12,18],'D10':[12,18],'D11':[12,18],'D12':[12,18],'H':[13,19],'O':[14,19],'T':[15,19], 'I':[16,19]}

# create inverse map
label_to_symm_map = {v: k for k, v in symm_to_label_map.items()}
coarse_label_to_symm_map = {v: k for k, v in symm_to_coarselabel_map.items()}
joint_label_to_symm_map = label_to_symm_map.copy()
joint_label_to_symm_map[17]='CX'
joint_label_to_symm_map[18]='DX'
joint_label_to_symm_map[19]='HOTI'
joint_label_to_symm_map[20]='Other'
label_to_symm_map[17] = 'Other'


if len(sys.argv) < 3:
    print('Usage: python analyze_res.py <predictions.pkl> <output-file-prefix>\n')
    sys.exit(0)


preds_file=sys.argv[1]
output_prefix=sys.argv[2]
granularity = 3

with open(preds_file,'rb') as fin:
   d=pickle.load(fin)

res=[]
for i in range(len(d)):
   res.append(d[i][1])

res=np.row_stack(res)
res.shape
y_pred = res

res=[]
for i in range(len(d)):
   res.append(d[i][2])

y_true=np.row_stack(res)

if y_true.dtype == np.float64:
    # Convert to integers
    y_true = y_true.astype(np.int64)

res=[]
for i in range(len(d)):
   res=res+d[i][0]
pdbids = res

print('Size of dataset: ',y_true.shape,len(pdbids))
print(np.sum(y_true,axis=0))

aucpr = metrics.average_precision_score(y_true=y_true, y_score=y_pred,average=None)
print('Avg AUC-PR:',np.mean(aucpr),'size:',y_pred.shape,'[sklearn] Average Precision Score:\n',aucpr)

if granularity == COARSE:
    label_map = coarse_label_to_symm_map
elif granularity == FINE:
    label_map = label_to_symm_map
else:
    label_map = joint_label_to_symm_map

labels_in_this_split = [label_map[l] for l in range(y_true.shape[1])]
y_pred_bin = (torch.sigmoid(torch.tensor(y_pred)) >= 0.5).float()  
report = metrics.classification_report(y_true=y_true, y_pred=y_pred_bin, target_names=labels_in_this_split, zero_division=0)
print('[sklearn] Performance:','\n',report)

## get confusion matrix on single label examples
y_true=y_true[:,fine_class_idxes]
y_pred=y_pred[:,fine_class_idxes]

idxes = np.where(np.sum(y_true,axis=1)==1)[0]

y_true_cnf = y_true[idxes,:]
y_pred_cnf = y_pred[idxes,:]

print('Single labeled proteins:',np.sum(y_true_cnf,axis=0))

y_pred_cnf = torch.argmax(torch.softmax(torch.tensor(y_pred_cnf), dim=1),dim=1)
y_true_cnf = y_true_cnf.argmax(axis=1)

counts = np.bincount(y_true_cnf)

print('Examples with only one label: ',y_true_cnf.shape, counts)
cnf = metrics.confusion_matrix(y_true=y_true_cnf, y_pred=y_pred_cnf) 
print(cnf)

y_pred = (torch.sigmoid(torch.tensor(y_pred)) >= 0.5).float()  
incorrect_predictions = np.logical_and(y_true == 0, y_pred == 1).sum(axis=0)
print('Overpredicted counts: ', incorrect_predictions)
results = {'confusion_matrix':cnf, 'incorrect_preds':incorrect_predictions, 'y_pred_cnf':y_pred_cnf, 'y_true_cnf':y_true_cnf}
with open(f'{output_prefix}_sklearn_metrics.pkl', 'wb') as fout:
    pickle.dump(results, fout)


