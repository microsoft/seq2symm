# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from __future__ import print_function
from __future__ import division
import argparse
import math
from sklearn import metrics
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import random
import os, numpy as np
import pickle

import esm

from torch import Tensor

from data_loader import CoarseNFineJointDataLoader, SimpleHomomerDataLoader, label_to_symm_map, coarse_label_to_symm_map, joint_label_to_symm_map, COARSE, FINE

from torch.utils import data
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import LambdaLR
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

from typing import Any, cast

from lightning.pytorch import LightningModule
from lightning.pytorch import Trainer
from lightning.pytorch.strategies import DDPStrategy
from torchmetrics import MetricCollection
from torchmetrics.classification import Precision, Recall, AveragePrecision, ConfusionMatrix, ROC 


from lightning.pytorch.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger


N_PRINT_TRAIN = 500
CACHE_EMPTY_FREQ = 20
ESM2_EMBEDDINGS_SIZE = 1280

# Generate a random seed for each run
random_seed = random.randint(0, 2**32 - 1)
## fix a seed
seed = 12306


random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Current device: ' + str(device),flush=True)


def add_weight_decay(model, l2_coeff):
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        print('Optimizing: ',name)
        if "norm" in name or name.endswith(".bias"):
            no_decay.append(param)
        else:
            decay.append(param)
    return [{'params': no_decay, 'weight_decay': 0.0}, {'params': decay, 'weight_decay': l2_coeff}]


def get_stepwise_decay_schedule_with_warmup(optimizer, num_warmup_steps, num_steps_decay, decay_rate, last_epoch=-1):
    """
    Create a schedule with a learning rate that decreases linearly from the initial lr set in the optimizer to 0, after
    a warmup period during which it increases linearly from 0 to the initial lr set in the optimizer.

    Args:
        optimizer (:class:`~torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (:obj:`int`):
            The number of steps for the warmup phase.
        num_training_steps (:obj:`int`):
            The total number of training steps.
        last_epoch (:obj:`int`, `optional`, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))

        num_fades = (current_step-num_warmup_steps)//num_steps_decay
        return (decay_rate**num_fades)

    return LambdaLR(optimizer, lr_lambda, last_epoch)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_parameters(model, only_optimized=False):
    for name, param in model.named_parameters():
        if only_optimized and not param.requires_grad:
            continue
        else:
            print(name)


# pytorch issue with dataloader that makes a list of tuples to list of lists.. so converting it back here
def conv_to_tuples(inlist):
    return list(map(lambda x: tuple((x[0][0],x[1][0])), inlist))


def entropy_minimization_loss(predictions, temperature=1.0):
    # Apply softmax to the predictions
    softmax_preds = F.softmax(predictions / temperature, dim=1)
    # Calculate the negative log likelihood loss
    loss = -torch.mean(torch.sum(softmax_preds * torch.log(softmax_preds + 1e-10), dim=1))  
    return loss


def gelu(x):
    """Implementation of the gelu activation function.

    For information: OpenAI GPT's gelu is slightly different
    (and gives slightly different results):
    0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


try:
    from apex.normalization import FusedLayerNorm as _FusedLayerNorm

    class ESM1bLayerNorm(_FusedLayerNorm):
        @torch.jit.unused
        def forward(self, x):
            if not x.is_cuda:
                return super().forward(x)
            else:
                with torch.cuda.device(x.device):
                    return super().forward(x)

except ImportError:
    from torch.nn import LayerNorm as ESM1bLayerNorm


class MultiLabelRankingLossWithIndicatorTarget(nn.Module):
    def __init__(self, margin=1.0):
        super(MultiLabelRankingLossWithIndicatorTarget, self).__init__()
        self.margin = margin
    def forward(self, logits, target):
        # Apply sigmoid to logits to get probabilities
        probs = torch.sigmoid(logits)
        # Calculate ranking loss for each sample
        ranking_loss = 0
        for i in range(logits.shape[0]):
            # Indices of positive and negative labels
            positive_indices = target[i].nonzero().view(-1)
            negative_indices = (1 - target[i]).nonzero().view(-1)         
            positive_vals = [probs[i,p] for p in positive_indices]
            negative_vals = [probs[i,n] for n in negative_indices]
            # Calculate pairwise ranking loss for positive and negative pairs
            pairs = torch.tensor(list(product(positive_vals, negative_vals)))  
            ranking_loss += F.margin_ranking_loss(input1=pairs[:,0],  ## max(0, input1-input2)
                                                input2=pairs[:,1], 
                                                target=torch.ones(pairs.shape[0]), margin=self.margin)
        # Average the ranking loss over the batch
        ranking_loss /= logits.shape[0]
        return ranking_loss


from itertools import product

class MultiLabelRankingLossWithSoftTarget(nn.Module):
    def __init__(self, margin=1.0):
        super(MultiLabelRankingLossWithSoftTarget, self).__init__()
        self.margin = margin
    def forward(self, logits, target):
        n_classes = target.shape[1]
        assert (n_classes > 1)
        # Apply sigmoid to logits to get probabilities
        probs = torch.sigmoid(logits)
        # Calculate ranking loss for each sample
        ranking_loss = 0
        for i in range(logits.shape[0]):            
            # Generate all pairs of elements
            pairs = list(product(range(n_classes), repeat=2))
            # Filter pairs where x > y
            valid_pairs = torch.tensor([(probs[i, x], probs[i, y]) for x, y in pairs if target[i][x] > target[i][y]])
            # Calculate pairwise ranking loss
            ranking_loss += F.margin_ranking_loss(input1=valid_pairs[:,0], 
                                                input2=valid_pairs[:,1], 
                                                target=torch.ones(valid_pairs.shape[0]), margin=self.margin)
        # Average the ranking loss over the batch
        ranking_loss /= logits.shape[0]
        return ranking_loss


class RobertaLMHead(nn.Module):
    """Head for masked language modeling."""

    def __init__(self, embed_dim, hidden_dim, output_dim, dropout_rate=0.1):
        super().__init__()
        self.dense1 = nn.Linear(embed_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)  # Dropout layer added
        self.layer_norm = ESM1bLayerNorm(hidden_dim)
        self.dense2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, features):   ## ESM2 features size: torch.Size([1, N, 1280])
        x = self.dense1(features)
        x = gelu(x)
        x = self.dropout(x)  # Dropout applied after activation
        x = self.layer_norm(x)
        # average over residues in protein.. 
        x = x.mean(1)
        x = self.dense2(x)
        return x


class RobertaLMHeadDeeper(nn.Module):
    """Head for masked language modeling.
     Example usage:
     esm2_model.lm_head = RobertaLMHeadDeeper(embed_dim=ESM2_EMBEDDINGS_SIZE, hidden_dim=[256,50],
          output_dim=self.n_classes, dropout_rate=0.2)        
    #"""        

    def __init__(self, embed_dim, hidden_dim, output_dim, dropout_rate=0.1):
        super().__init__()
        self.dense1 = nn.Linear(embed_dim, hidden_dim[0])
        self.dropout = nn.Dropout(dropout_rate)  # Dropout layer added
        self.layer_norm1 = ESM1bLayerNorm(hidden_dim[0])
        self.dense2 = nn.Linear(hidden_dim[0], hidden_dim[1])
        self.layer_norm2 = ESM1bLayerNorm(hidden_dim[1])
        self.dense3 = nn.Linear(hidden_dim[1], output_dim)        

    def forward(self, features):   ## ESM2 features size: torch.Size([1, N, 1280])
        x = self.dense1(features)
        x = gelu(x)
        x = self.dropout(x)  # Dropout applied after activation
        x = self.layer_norm1(x)
        # average over residues in protein.. 
        x = x.mean(1)
        x = self.dense2(x)
        x = gelu(x)
        x = self.dropout(x)  # Dropout applied after activation
        x = self.layer_norm2(x)
        x = self.dense3(x)
        return x



class MultiheadedRobertaLMHead(nn.Module):
    def __init__(self,
                    embed_dim,
                    hidden_layer_sizes=[256, 100],
                    output_dim=20):
        
        super().__init__()

        n_classes = output_dim
        self.common_head = RobertaLMHead(embed_dim, hidden_layer_sizes[0], hidden_layer_sizes[1])

        # fine heads
        self.fine_heads_one_per_class = []
        for cl in range(n_classes):
            self.head2_fc1 = nn.Linear(hidden_layer_sizes[1], 1)
            self.fine_heads_one_per_class.append(self.head2_fc1)
        self.fine_heads_one_per_class = nn.ModuleList(self.fine_heads_one_per_class)

        self.layer_norm1 = ESM1bLayerNorm(hidden_layer_sizes[1])


    def forward(self, features):
        x = self.common_head(features)     
        x = self.layer_norm1(gelu(x))

        self.fine_heads_one_per_class = self.fine_heads_one_per_class.to(features.device)
        out = torch.cat([self.fine_heads_one_per_class[head](x) for head in range(len(self.fine_heads_one_per_class))], dim=1)

        return out


class MultiheadedNoSharedRobertaLMHead(nn.Module):
    """
    Example usage:
    esm2_model.lm_head = MultiheadedNoSharedRobertaLMHead(
        embed_dim=ESM2_EMBEDDINGS_SIZE, hidden_layer_sizes=[256], 
        output_dim=self.n_classes, dropout_rate=0.2)  
    """
    def __init__(self,
                    embed_dim,
                    hidden_layer_sizes=[256],
                    output_dim=20, dropout_rate=0.2):
        
        super().__init__()

        n_classes = output_dim

        # fine heads
        self.fine_heads_one_per_class = []
        for cl in range(n_classes):
            self.head = RobertaLMHead(embed_dim=embed_dim, hidden_dim=hidden_layer_sizes[0], output_dim=1, dropout_rate=dropout_rate)
            self.fine_heads_one_per_class.append(self.head)
        self.fine_heads_one_per_class = nn.ModuleList(self.fine_heads_one_per_class)

    def forward(self, features):
        self.fine_heads_one_per_class = self.fine_heads_one_per_class.to(features.device)
        out = torch.cat([self.fine_heads_one_per_class[head](features) for head in range(len(self.fine_heads_one_per_class))], dim=1)

        return out



class ESMFinetuner(LightningModule):
    ## params: n_classes=17, n_epoch=100, lr=1.0e-4, l2_coeff=1.0e-2, interactive=False):
    def __init__(self, params, **kwargs: Any) -> None:  

        super().__init__()

        self.params = params
        if hasattr(params, 'test_mode') and not(params.test_mode):
            self.n_classes = params.n_classes
            self.model_dir = params.model_dir
            self.output_model = params.output_model
            self.n_epoch = params.n_epoch
            self.init_lr = params.lr
            self.l2_coeff = params.l2_coeff
            self.port = params.port
            self.ACCUM_STEP = 1
            self.batch_size = params.bs
            print('Batch size: ',self.batch_size)
            self.model = self.config_model(params.num_layers_frozen)
            self.set_loss_and_metrics()
        else:
            self.n_classes = params.n_classes
            self.n_epoch = params.n_epoch
            self.batch_size = params.bs
            self.model = self.config_model(params.num_layers_frozen)


        """## HACK TO GET IT TO PREDICT ON TRAINING DATA
        params.weighted_sampler = 0
        self.params.weighted_sampler = 0
        self.params.batch_size = 48
        params.bs = 48

        #self.set_loss_and_metrics()
        ############## """


    def set_loss_and_metrics(self):
        if self.params.use_margin_loss:
            if self.params.use_soft_labels:
                self.loss = nn.CrossEntropyLoss() 
                self.margin_loss = MultiLabelRankingLossWithSoftTarget()
            else:
                self.loss = nn.BCEWithLogitsLoss()        
                self.margin_loss = MultiLabelRankingLossWithIndicatorTarget()
        else:
            self.loss = nn.BCEWithLogitsLoss()        

        self.train_metrics_set1 = MetricCollection(
            [
                AveragePrecision(task="multilabel", num_labels=self.n_classes, num_classes=self.n_classes, average='none')              
            ],
            prefix="train_set1",
        )
        self.train_metrics_set2 = MetricCollection(
            [
                Precision(task="multilabel", num_labels=self.n_classes, num_classes=self.n_classes, average='none'),
                Recall(task="multilabel", num_labels=self.n_classes, num_classes=self.n_classes, average='none')
            ],
            prefix="train_set2",
        )

        self.val_metrics_set1 = self.train_metrics_set1.clone(prefix="val_set1")
        self.val_metrics_set2 = self.train_metrics_set2.clone(prefix="val_set2")
        self.test_metrics_set1 = self.train_metrics_set1.clone(prefix="test_set1")
        self.test_metrics_set2 = self.train_metrics_set2.clone(prefix="test_set2")       
        self.save_hyperparameters()


    def config_model(self, num_layers_frozen):
        esm2_model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        esm2_model.lm_head = RobertaLMHead(embed_dim=ESM2_EMBEDDINGS_SIZE, hidden_dim=256,
            output_dim=self.n_classes, dropout_rate=0.2)

        # unfreeze some params
        for param in esm2_model.embed_tokens.parameters():
            param.requires_grad = False
        for param in esm2_model.contact_head.parameters():
            param.requires_grad = False
        for param in esm2_model.emb_layer_norm_after.parameters():
            param.requires_grad = False
        for ii in range(num_layers_frozen):
            for param in esm2_model.layers[ii].parameters():
                param.requires_grad = False

        print('Number of parameters being optimized: ',count_parameters(esm2_model))
        self.batch_converter = alphabet.get_batch_converter()
        return esm2_model


    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Forward pass of the model.
        Args:
            x: tensor of data to run through the model
        Returns:
            output from the model
        """
        return self.model(*args, **kwargs)
    

    def configure_optimizers(self):
        """Initialize the optimizer and learning rate scheduler.
        Returns:
            learning rate dictionary
        """
        opt_params = add_weight_decay(self.model, self.l2_coeff)
        optimizer = torch.optim.AdamW(opt_params, lr=self.init_lr)

        return {
                    "optimizer": optimizer
        }


    def training_step(self, batch, batch_idx):
        """Compute and return the training loss.
        Args:
            batch: the output of your DataLoader
        Returns:
            training loss
        """
        x = batch["x"]
        y = batch["y"]

        x = x.to(next(self.model.parameters()).device)

        output = self(x)
        logits_homomer = output['logits']
        loss = self.loss(logits_homomer, y)

        if self.params.use_margin_loss:
            loss2 = self.margin_loss(logits_homomer, y)
            loss = loss + loss2

        if self.params.use_soft_labels:
            y_soft = y.clone()
            y = (y>0).int()

        # compute first set of metrics that use logits
        self.train_metrics_set1(logits_homomer, y.int())
        # Convert logits to per-class probability using sigmoid for other metrics
        probs = torch.sigmoid(logits_homomer)
        # Threshold probabilities to obtain binary predictions (0 or 1)
        pred_labels = cast(Tensor, (probs >= 0.5).float())
        # compute second set that uses predictions
        self.train_metrics_set2(pred_labels, y.int())        

        # by default, the train step logs every `log_every_n_steps` steps where
        # `log_every_n_steps` is a parameter to the `Trainer` object
        self.log("training_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        if batch_idx % N_PRINT_TRAIN == 0:
            print("\n[LOG] Epoch:",self.current_epoch," Batch:",batch_idx,"training_loss",loss.cpu().detach().numpy())

        return loss


    # changing each element from a list of two tuples to a single tuple
    def conv_to_tuples(self, inlist):
        return list(map(lambda x: tuple((x[0][0],x[1][0])), inlist))
 

    def validation_step(self, batch, batch_idx):
        """Compute validation loss.
        Args:
            batch: the output of your DataLoader
        """
        x = batch["x"]
        y = batch["y"]
        
        x = x.to(next(self.model.parameters()).device)

        output = self(x)
        logits_homomer = output['logits']
        loss = self.loss(logits_homomer, y)

        if self.params.use_margin_loss:
            loss2 = self.margin_loss(logits_homomer, y)
            loss = loss + loss2        

        if self.params.use_soft_labels:
            y_soft = y.clone()
            y = (y>0).int()

        self.val_metrics_set1(logits_homomer, y.int())
        # Convert logits to per-class probabilities using sigmoid activation
        probs = torch.sigmoid(logits_homomer)
        # Threshold probabilities to obtain binary predictions (0 or 1)
        pred_labels = cast(Tensor, (probs >= 0.5).float())

        # by default, the test and validation steps only log per *epoch*
        self.log("validation_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        if batch_idx % (N_PRINT_TRAIN) == 0:
            print("\n[LOG] Epoch:",self.current_epoch," Batch:",batch_idx,"validation_loss",loss.cpu().detach().numpy())
        self.val_metrics_set2(pred_labels, y.int())

        return loss


    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        """Compute test loss.
        Args:
            batch: the output of your DataLoader
        """
        x = batch["x"]
        y = batch["y"]

        x = x.to(next(self.model.parameters()).device)

        output = self(x)
        logits_homomer = output['logits']
        loss = self.loss(logits_homomer, y)
        if self.params.use_margin_loss:
            loss2 = self.margin_loss(logits_homomer, y)
            loss = loss + loss2        

        if self.params.use_soft_labels:
            y_soft = y.clone()
            y = (y>0).int()        

        self.test_metrics_set1(logits_homomer, y.int())
        # Convert logits to probabilities using sigmoid activation
        probs = torch.sigmoid(logits_homomer)
        # Threshold probabilities to obtain binary predictions (0 or 1)
        pred_labels = cast(Tensor, (probs >= 0.5).float())

        # by default, the test and validation steps only log per *epoch*
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        if batch_idx % (N_PRINT_TRAIN) == 0:
            print("\n[LOG] Epoch:",self.current_epoch," Batch:",batch_idx,"test_loss",loss.cpu().detach().numpy())        
        self.test_metrics_set2(pred_labels, y.int())

        return loss      


    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x = batch["x"]
        y = batch["y"]
        pdbids = batch["pdbids"]
        
        x = x.to(next(self.model.parameters()).device)

        output = self(x)
        logits_homomer = output['logits']

        """ Uncomment the code below to get embeddings
        output = self(x, repr_layers=[31])
        logits_homomer = output['representations'][31]
        logits_homomer = logits_homomer.mean(1)
        """

        return pdbids, logits_homomer, y

   
    def save_results(self, epoch, metrics_dict, prefix):
        res = np.concatenate([
            val.cpu().detach().numpy().tolist()
            for val in metrics_dict.values()
        ]).tolist()
        res = [epoch] + res
        
        with open(format("%s/%s_results.csv" % (self.params.output_dir, prefix)),'a') as fin:
            fin.write(",".join([str(x) for x in res]))
            fin.write('\n')
            fin.close()


    @torch.no_grad()
    def on_test_epoch_end(self) -> None:
        """Logs epoch level test metrics."""      
        metrics_dict = self.test_metrics_set1.compute()
        self.test_metrics_set2.compute()
        for (m, val) in metrics_dict.items():
            print(m,val)            
            print('\n[LOG] Test performance: ',m, ' Avg perf:',val.nanmean().cpu().detach().numpy(),
                  'Class-wise:'," ".join([str(x) for x in val.cpu().detach().numpy().tolist()]))
            self.log(m, val.nanmean(), sync_dist=True)

        self.test_metrics_set1.reset()
        self.test_metrics_set2.reset()


    def on_train_epoch_end(self) -> None:
        """Logs epoch level training metrics."""       
        metrics_dict = self.train_metrics_set1.compute()
        self.train_metrics_set2.compute()
        for (m, val) in metrics_dict.items():
            print(m,val)
            print('\n[LOG] Train performance: ',m, ' Avg perf:',val.nanmean().cpu().detach().numpy(),
                  'Class-wise:'," ".join([str(x) for x in val.cpu().detach().numpy().tolist()]))
            self.log(m, val.nanmean(), sync_dist=True)

        self.train_metrics_set1.reset()
        self.train_metrics_set2.reset()


    def on_validation_epoch_end(self) -> None:
        """Logs epoch level validation metrics."""      
        metrics_dict = self.val_metrics_set1.compute()
        self.val_metrics_set2.compute()
        for (m, val) in metrics_dict.items():
            print(m,val)            
            print('\n[LOG] Validation performance: ',m, ' Avg perf:',val.nanmean().cpu().detach().numpy(),
                  'Class-wise:'," ".join([str(x) for x in val.cpu().detach().numpy().tolist()]))
            self.log(m, val.nanmean(), sync_dist=True)    
        
        self.save_results(self.current_epoch, metrics_dict, 'validation')
        self.val_metrics_set1.reset()
        self.val_metrics_set2.reset()


def compute_sklearn_metrics_old(list_of_tuples, splitid, granularity, out_dir):
    y_true = []
    y_pred = []
    for pred, label in list_of_tuples:
        y_true.append(label)
        y_pred.append(pred)
    y_pred = torch.row_stack(y_pred)
    y_true = torch.row_stack(y_true)
    aucpr = metrics.average_precision_score(y_true=y_true, y_score=y_pred,average=None)
    print(splitid,'size:',y_pred.shape,'[sklearn] Average Precision Score:\n',aucpr)
    aucpr = 0.0
    if granularity == COARSE:
        label_map = coarse_label_to_symm_map
    elif granularity == FINE:
        label_map = label_to_symm_map
    else:
        label_map = joint_label_to_symm_map
    labels_in_this_split = [label_map[l] for l in range(y_true.shape[1])]
    y_pred = (torch.sigmoid(y_pred) >= 0.5).float()  
    report = metrics.classification_report(y_true=y_true, y_pred=y_pred, target_names=labels_in_this_split, zero_division=0)
    print('[sklearn] Performance on:',splitid,'\n',report)
    cnf = metrics.multilabel_confusion_matrix(y_true=y_true, y_pred=y_pred, labels=labels_in_this_split)

    results = {'aucpr':aucpr, 'prec_rec_report':report, 'confusion_matrix':cnf}
    with open(f'{out_dir}/sklearn_metrics.pkl', 'wb') as fout:
        pickle.dump(results, fout)


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

    if params.use_soft_labels:
        y_soft = y_true.clone()
        y_true = (y_true>0).int()

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



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help='path to training directory')
    parser.add_argument('--model_dir', type=str, required=True, help='model directory name')
    parser.add_argument('--output_model', type=str, required=False, help='output prefix for model')
    parser.add_argument('--output_dir', type=str, required=True, help='output directory to save predictions')
    parser.add_argument('--meta_data_file', type=str, required=True, help='path to metadata file')
    parser.add_argument('--bs', type=int, default=2, required=False, help='batch size')
    parser.add_argument('--data_splits_file', type=str, required=True, help='file containing training, validation and test clusterid splits')
    parser.add_argument('--lr', type=float, default=1e-5, required=False, help='learning rate')
    parser.add_argument('--l2_coeff', type=float, default=1e-1, required=False, help='l2 regularization coeff')
    parser.add_argument('--n_classes', type=int, default=18, required=False, help='number of classes')
    parser.add_argument('--n_epoch', type=int, default=2, required=False, help='number of epochs to train for')
    parser.add_argument('--port', type=int, default=12345, required=False, help='port number')
    parser.add_argument('--limit', type=int, default=65536, required=False, help='limit size of dataset per gpu')
    parser.add_argument('--weighted_sampler', type=int, default=0, required=False, help='use weighted sampler for training [0=None, 1=cluster-size-based]')
    parser.add_argument('--num_layers_frozen', type=int, default=31, required=False, help='number of layers to freeze in ESM2')
    parser.add_argument('--granularity', type=int, required=True, default=2, help='coarse=1 fine=2 both=3')    
    parser.add_argument('--use_soft_labels', type=int, required=False, default=0, help='use soft labels or not')    
    parser.add_argument('--use_margin_loss', type=int, required=False, default=0, help='use margin loss')    
    parser.add_argument('--checkpoint_path', type=str, required=False, help='path to checkpoint file')
    
    start_time = time.time()
    
    params = parser.parse_args()
    print(params,flush=True)
    
    task = ESMFinetuner(params=params)
    dataloader = CoarseNFineJointDataLoader(params=params, collater=task.batch_converter)
    checkpoint_callback = ModelCheckpoint(dirpath=format('%s/%s' % (params.model_dir,params.output_model)), save_top_k=-1, auto_insert_metric_name=True, monitor='validation_loss')

    logger = CSVLogger(params.output_dir, name=params.output_model)
    trainer = Trainer(accelerator="cuda", strategy=DDPStrategy(find_unused_parameters=True),
                      max_epochs=params.n_epoch, limit_train_batches=1.0,
                      default_root_dir=format('%s/%s' % (params.model_dir,params.output_model)), callbacks=[checkpoint_callback],
                      accumulate_grad_batches=2,
                      val_check_interval=0.5,
                      gradient_clip_val=0.2,
                      logger=logger)
    if params.checkpoint_path:
        print('Reading checkpoint from..',params.checkpoint_path)
        trainer.fit(model=task, datamodule=dataloader,ckpt_path=params.checkpoint_path)
    else:
        trainer.fit(model=task, datamodule=dataloader)  


    predictions = trainer.predict(model=task, dataloaders=[dataloader.val_dataloader()])
    with open(format('%s/val_predictions.pkl' % params.output_dir), 'wb') as fout:
        pickle.dump(predictions, fout)
    compute_sklearn_metrics(predictions,'validation',params)        

    predictions = trainer.predict(model=task, dataloaders=[dataloader.test_dataloader()])
    with open(format('%s/test_predictions.pkl' % params.output_dir), 'wb') as fout:
        pickle.dump(predictions, fout)
    compute_sklearn_metrics(predictions,'test',params)

    print('Total time taken: ',time.time()-start_time,flush=True)


