# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np
import pickle
import sys
import torch
from sklearn import metrics

preds_file=sys.argv[1]
output_prefix=sys.argv[2]
epoch=sys.argv[3]

with open(preds_file,'rb') as fin:
   d=pickle.load(fin)

res=[]
for i in range(len(d)):
   res.append(d[i][1])

res=np.row_stack(res)
print(res.shape)
y_pred = res

res=[]
for i in range(len(d)):
   res.append(d[i][2])

y_true=np.row_stack(res)
print(y_true.shape)

if y_true.dtype == np.float64:
    # Convert to integers
    y_true = y_true.astype(np.int64)

res=[]
for i in range(len(d)):
   res.append(d[i][0])
   
pdbids = sum(res, [])

print('Size of dataset: ',y_true.shape,len(pdbids))

predictions = {'labels': y_true, 'pdb_ids': pdbids, 'logits': y_pred}

with open(format('%s_predictions_all_epoch%s.pkl' % (output_prefix,epoch)),'wb') as fout:
    pickle.dump(predictions,fout)

