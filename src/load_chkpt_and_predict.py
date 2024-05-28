# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from __future__ import print_function
from __future__ import division
import argparse
from sklearn import metrics
import torch
import time
import random
import numpy as np
import pickle

from data_loader import CoarseNFineJointDataLoader, TestSequencesLoader, label_to_symm_map, coarse_label_to_symm_map, joint_label_to_symm_map, COARSE, FINE, SimpleHomomerDataLoader

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

from lightning.pytorch import Trainer
from lightning.pytorch.strategies import DDPStrategy

from lightning.pytorch.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

from finetune import ESMFinetuner

seed = 12306
random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Current device: ' + str(device),flush=True)


def compute_sklearn_metrics(list_of_tuples, splitid, params):
    y_true = []
    y_pred = []
    y_true_cnf = []
    y_pred_cnf = []    
    for pdbid, pred, label in list_of_tuples:
        y_true.append(label)
        y_pred.append(pred)
        if torch.sum(label) == 1:
            y_true_cnf.append(label)
            y_pred_cnf.append(pred)
    y_pred = torch.row_stack(y_pred)
    y_true = torch.row_stack(y_true)
    aucpr = metrics.average_precision_score(y_true=y_true, y_score=y_pred,average=None)
    print(splitid,'size:',y_pred.shape,'[sklearn] Average Precision Score:\n',aucpr)
    aucpr = 0.0
    if params.granularity == COARSE:
        label_map = coarse_label_to_symm_map
    elif params.granularity == FINE:
        label_map = label_to_symm_map
    else:
        label_map = joint_label_to_symm_map
    labels_in_this_split = [label_map[l] for l in range(y_true.shape[1])]
    y_pred = (torch.sigmoid(y_pred) >= 0.5).float()
    report = metrics.classification_report(y_true=y_true, y_pred=y_pred, target_names=labels_in_this_split, zero_division=0)
    print('[sklearn] Performance on:',splitid,'\n',report)
    ## confusion only on single label categories
    y_pred_cnf = torch.row_stack(y_pred_cnf)
    y_true_cnf = torch.row_stack(y_true_cnf).argmax(dim=1)    
    y_pred_cnf = torch.argmax(torch.softmax(y_pred_cnf, dim=1),dim=1)
    y_pred_cnf = torch.nn.functional.one_hot(y_pred_cnf, num_classes=y_true.shape[1]).argmax(dim=1)
    labels_in_this_split = [label_map[int(l)] for l in y_true_cnf]
    counts = torch.bincount(y_true_cnf)
    print('Examples with only one label: ',y_true_cnf.shape, counts)
    cnf = metrics.confusion_matrix(y_true=y_true_cnf, y_pred=y_pred_cnf) 
    print(cnf)
    incorrect_predictions = np.logical_and(y_true == 0, y_pred == 1).sum(dim=0)
    print('Overpredicted counts: ', incorrect_predictions)
    results = {'aucpr':aucpr, 'prec_rec_report':report, 'confusion_matrix':cnf, 'incorrect_preds':incorrect_predictions}
    with open(f'{params.output_dir}/{splitid}_sklearn_metrics.pkl', 'wb') as fout:
        pickle.dump(results, fout)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help='path to training directory')
    parser.add_argument('--model_dir', type=str, required=True, help='model directory name')
    parser.add_argument('--output_model', type=str, required=False, help='output prefix for model')
    parser.add_argument('--output_dir', type=str, required=True, help='output directory to save predictions')
    parser.add_argument('--meta_data_file', type=str, required=False, default=None, help='path to metadata file')
    parser.add_argument('--bs', type=int, default=2, required=False, help='batch size')
    parser.add_argument('--data_splits_file', type=str, required=False, default=None, help='file containing training, validation and test clusterid splits')
    parser.add_argument('--lr', type=float, default=1e-4, required=False, help='learning rate')
    parser.add_argument('--l2_coeff', type=float, default=1e-3, required=False, help='learning rate')
    parser.add_argument('--n_classes', type=int, default=17, required=False, help='number of classes')
    parser.add_argument('--n_epoch', type=int, default=2, required=False, help='number of epochs to train for')
    parser.add_argument('--port', type=int, default=12345, required=False, help='port number')
    parser.add_argument('--num_layers_frozen', type=int, default=31, required=False, help='number of layers to freeze in MSA-Transformer')
    parser.add_argument('--granularity', type=int, required=True, default=2, help='coarse=1 fine=2 both=3')    
    parser.add_argument('--weighted_sampler', type=int, default=0, required=False, help='use weighted sampler for training [0=None, 1=cluster-size-based]')
    parser.add_argument('--chkpt_file', type=str, required=True, help='checkpoint file name')
    parser.add_argument('--suffix', type=str, required=True, help='suffix to save predictions')
    parser.add_argument('--use_soft_labels', type=int, required=False, default=0, help='use soft labels or not')    
    parser.add_argument('--use_margin_loss', type=int, required=False, default=0, help='use margin loss')    

    
    start_time = time.time()
    
    params = parser.parse_args()
    print('PARAMS: ',params,flush=True)
    
    task = ESMFinetuner(params=params)

    task = task.load_from_checkpoint(params.chkpt_file)  
    task.eval()

    if params.data_splits_file:
        if params.n_classes <= 17:
            dataloader = SimpleHomomerDataLoader(params=params, collater=task.batch_converter)
        else:
            dataloader = CoarseNFineJointDataLoader(params=params, collater=task.batch_converter)
    else:
        dataloader = TestSequencesLoader(params=params, collater=task.batch_converter)

    checkpoint_callback = ModelCheckpoint(dirpath=params.model_dir, auto_insert_metric_name=True, monitor='validation_loss')

    logger = CSVLogger(params.output_dir, name=params.output_model)

    trainer = Trainer(accelerator="cuda", strategy=DDPStrategy(find_unused_parameters=True),
                      max_epochs=params.n_epoch,
                      default_root_dir=format('%s/%s' % (params.model_dir,params.output_model)), callbacks=[checkpoint_callback],
                      logger=logger)

    predictions = trainer.predict(model=task, dataloaders=[dataloader.val_dataloader()])
    with open(format('%s/val_predictions_%s.pkl' % (params.output_dir,params.suffix)), 'wb') as fout:
        pickle.dump(predictions, fout)
    compute_sklearn_metrics(predictions,'validation',params)            

    predictions = trainer.predict(model=task, dataloaders=[dataloader.test_dataloader()])
    with open(format('%s/test_predictions_%s.pkl' % (params.output_dir, params.suffix)), 'wb') as fout:
        pickle.dump(predictions, fout)
    compute_sklearn_metrics(predictions,'test',params)

    print('Total time taken: ',time.time()-start_time,flush=True)
    
