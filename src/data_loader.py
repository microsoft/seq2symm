# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from functools import partial
import gzip
import string
import traceback
from lightning import LightningDataModule
import pandas as pd
import torch
from torch.utils import data
from torch.utils.data import DataLoader, Dataset
import numpy as np
import torch.distributed as dist
from typing import List, Tuple

from Bio import SeqIO

import os, pickle
import linecache


from scipy.spatial.distance import cdist


symm_to_label_map = {'C1':0, 'C2':1,'C3':2,'C4':3,'C5':4,'C6':5,'C7':6,'C8':6,'C9':6,'C10':7,'C11':7,'C12':7,'C13':7,'C14':7,
'C15':7,'C16':7,'C17':7,'D2':8,'D3':9,'D4':10,'D5':11,'D6':12,'D7':12,'D8':12,'D9':12,'D10':12,'D11':12,'D12':12,
'H':13,'O':14,'T':15, 'I':16}

symm_to_coarselabel_map = {'C1':0, 'C2':1, 'D2':2, 'C3':3,'C4':3,'C5':3,'C6':3,'C7':3,'C8':3,'C9':3,'C10':3,'C11':3,'C12':3,'C13':3,'C14':3,
'C15':3,'C16':3,'C17':3, 'D3':4,'D4':4,'D5':4,'D6':4,'D7':4,'D8':4,'D9':4,'D10':4,'D11':4,'D12':4,
'H':5,'O':5,'T':5, 'I':5}

symm_to_joint_label_map = {'C1':0, 'C2':1,'C3':[2,17],'C4':[3,17],'C5':[4,17],'C6':[5,17],'C7':[6,17],'C8':[6,17],'C9':[6,17],'C10':[7,17],
'C11':[7,17],'C12':[7,17],'C13':[7,17],'C14':[7,17],'C15':[7,17],'C16':[7,17],'C17':[7,17],'D2':8,'D3':[9,18],
'D4':[10,18],'D5':[11,18],'D6':[12,18],'D7':[12,18],'D8':[12,18],'D9':[12,18],'D10':[12,18],'D11':[12,18],'D12':[12,18],
'H':[13,19],'O':[14,19],'T':[15,19], 'I':[16,19]}


# create inverse map
label_to_symm_map = {v: k for k, v in symm_to_label_map.items()}
coarse_label_to_symm_map = {v: k for k, v in symm_to_coarselabel_map.items()}
joint_label_to_symm_map = label_to_symm_map.copy()
joint_label_to_symm_map[17]='CX'
joint_label_to_symm_map[18]='DX'
joint_label_to_symm_map[19]='HOTI'
joint_label_to_symm_map[20]='Other'

label_to_symm_map[17] = 'Other'

COARSE = 1
FINE = 2


# This is an efficient way to delete lowercase characters and insertion characters from a string
deletekeys = dict.fromkeys(string.ascii_lowercase)
deletekeys["."] = None
deletekeys["*"] = None
translation = str.maketrans(deletekeys)


def read_fasta(filename: str) -> Tuple[str, str]:
    """ Reads the first (reference) sequences from a fasta or MSA file."""
    record = next(SeqIO.parse(filename, "fasta"))
    return record.description, str(record.seq)

def read_fasta_many_seqs(filename: str) -> [Tuple[str, str]]:
    dataset = []
    for _, seq_record in enumerate(SeqIO.parse(filename, "fasta")):
        dataset.append((seq_record.description, str(seq_record.seq)))
    return dataset

def remove_insertions(sequence: str) -> str:
    """ Removes any insertions into the sequence. Needed to load aligned sequences in an MSA. """
    return sequence.translate(translation)


def get_train_valid_set_seq_split(params):
    # read & clean list.csv
    print('Reading: ',params.meta_data_file)
    df = pd.read_csv(params.meta_data_file,header=0,sep=',',dtype='str')

    # read train/valid/test splits
    print('Reading: ',params.data_splits_file)
    with open(params.data_splits_file, 'rb') as fin:
        splits = pickle.load(fin)

    # compile training and validation sets
    train_pdb_dict = {}
    valid_pdb_dict = {}
    test_pdb_dict = {}    
    for _, row in df.iterrows():
        if (int(row['CLUSTER']) in splits['train_clusterids']):
            train_pdb_dict[row['CHAINID']] = (row['CHAINID'], row['HASH'], row['CLUSTER'], row['SYMM'])
        if (int(row['CLUSTER']) in splits['validation_clusterids']) and (':' not in row['CHAINID']):
            valid_pdb_dict[row['CHAINID']] = (row['CHAINID'], row['HASH'], row['CLUSTER'], row['SYMM'])
        if (int(row['CLUSTER']) in splits['test_clusterids']) and (':' not in row['CHAINID']):
            test_pdb_dict[row['CHAINID']] = (row['CHAINID'], row['HASH'], row['CLUSTER'], row['SYMM'])

    print('[LOG] Train, Validation, Test set sizes: ',len(train_pdb_dict),len(valid_pdb_dict),len(test_pdb_dict))
    return train_pdb_dict, valid_pdb_dict, test_pdb_dict


def read_data(params):
    print('Reading: ',params.meta_data_file)
    df = pd.read_csv(params.meta_data_file,header=0,sep=',',dtype='str')
    clusterid_to_pdbids = {}
    for _, row in df.iterrows():
        if not (row['CLUSTER'] in clusterid_to_pdbids):
            clusterid_to_pdbids[row['CLUSTER']] = []
        clusterid_to_pdbids[row['CLUSTER']].append(row['CHAINID'])

    # read train/valid/test splits
    print('Reading: ',params.data_splits_file)
    with open(params.data_splits_file, 'rb') as fin:
        splits = pickle.load(fin)

    # compile training and validation sets
    train_pdb_dict = {}
    valid_pdb_dict = {}
    test_pdb_dict = {}    
    for _, row in df.iterrows():
        if (int(row['CLUSTER']) in splits['train_clusterids']):
            train_pdb_dict[row['CHAINID']] = (row['CHAINID'], row['HASH'], row['CLUSTER'], row['SYMM'], len(row['SEQUENCE']))
        if (int(row['CLUSTER']) in splits['validation_clusterids']) and (':' not in row['CHAINID']):
            valid_pdb_dict[row['CHAINID']] = (row['CHAINID'], row['HASH'], row['CLUSTER'], row['SYMM'], len(row['SEQUENCE']))
        if (int(row['CLUSTER']) in splits['test_clusterids']) and (':' not in row['CHAINID']):
            test_pdb_dict[row['CHAINID']] = (row['CHAINID'], row['HASH'], row['CLUSTER'], row['SYMM'], len(row['SEQUENCE']))

    print('[LOG] Train, Validation, Test set sizes: ',len(train_pdb_dict),len(valid_pdb_dict),len(test_pdb_dict))

    datadict = {"train_split": train_pdb_dict, "valid_split": valid_pdb_dict, 
                "test_split": test_pdb_dict, "clusterid_to_pdbids": clusterid_to_pdbids}
    
    return datadict

## Written especially for ESM-MSA
def parse_a3m(filename):
    msa = []
    table = str.maketrans(dict.fromkeys(string.ascii_lowercase))
    try:
        if filename.split('.')[-1] == 'gz':
            fp = gzip.open(filename, 'rt')
        else:
            fp = open(filename, 'r')
        msa_ids = []
        # read file line by line
        for line in fp:
            # skip labels
            if line[0] == '>':
                msa_id = line.strip().replace('>','')
                msa_ids.append(msa_id)
                continue
            # remove right whitespaces
            line = line.rstrip()
            if len(line) == 0:
                continue
            # remove lowercase letters and append to MSA
            msa.append(line.translate(table))
            # sequence length
            L = len(msa[-1])
            # 0 - match or gap; 1 - insertion
            a = np.array([0 if c.isupper() or c=='-' else 1 for c in line])
            i = np.zeros((L))
            if np.sum(a) > 0:
                # positions of insertions
                pos = np.where(a==1)[0]
                # shift by occurrence
                a = pos - np.arange(pos.shape[0])
                # position of insertions in cleaned sequence
                # and their length
                pos,num = np.unique(a, return_counts=True)
                # append to the matrix of insetions
                i[pos] = num
            if len(msa) == 10000:
                break
    except Exception as e:
        print('[LOG] File reading error for: ',filename)            
        traceback.print_exception(type(e), e, e.__traceback__)  
        return (None, None)    
    return (msa_ids[0], msa[0])

def fasta_loader_old(item, params):
    fasta = get_fasta(params.data_dir + '/a3m/' + item[1][:3] + '/' + item[1] + '.a3m.gz')    
    return fasta


# item is a tuple: ('PDBID', 'HASH')  or ('PDBID', 'FILEPATH')
def fasta_loader(item, params): 
    if hasattr(params, 'test_mode') and params.test_mode:
        return read_fasta(item[1])
    else:
        if item[0].startswith('UniRef50'):
            return read_fasta(params.data_dir + '/fastas/' + item[1][0:2] + '/' + item[1][2:] + '/' + item[0] + '.fasta')
        else:
            return read_fasta(params.data_dir + '/fastas/' + item[0][1:3] + '/' + item[0].split(':')[0] + '.fasta')   


def get_fasta(a3mfilename):
    msa = parse_a3m(a3mfilename)
    return msa

def one_hot_encode(label, num_classes):
    """
    Converts a label to a one-hot encoding vector.

    Args:
        label (int): The label to be encoded.
        num_classes (int): The total number of classes.

    Returns:
        one_hot (list): A one-hot encoding vector.
    """
    one_hot = np.zeros((num_classes))
    one_hot[label] = 1
    return one_hot


# Load PDB examples
# item is a tuple: ('PDBID', 'HASH', 'CLUSTER', 'SYMM', 'LENGTH')
def loader_pdb(item, params):  
    msa = parse_a3m(params.data_dir + '/a3m/' + item[1][:3] + '/' + item[1] + '.a3m.gz')
    labels = item[3].split(' ')
    homomer_label_vecs = []
    # map labels to integers and one-hot encode them
    if params.granularity == COARSE:
        local_map = symm_to_coarselabel_map
    else:
        local_map = symm_to_label_map

    for l in labels:
        if (l in local_map):
            homomer_label = local_map[l]
        else:
            homomer_label = params.n_classes-1
        homomer_label_vecs.append(one_hot_encode(homomer_label, num_classes=params.n_classes))
    # address multilabel cases
    if len(homomer_label_vecs) > 1:
        homomer_label_vecs = np.row_stack(homomer_label_vecs)
        homomer_label_vecs = np.sum(homomer_label_vecs,axis=0)
    else:
        homomer_label_vecs = homomer_label_vecs[0]

    msa = greedy_select(msa, num_seqs=128) # can change this to pass more/fewer sequences    

    data = {"msa":msa, "label": homomer_label_vecs, "pdbid": item[0]}

    return data



def greedy_select(msa: List[Tuple[str, str]], num_seqs: int, mode: str = "max") -> List[Tuple[str, str]]:
    assert mode in ("max", "min")
    if len(msa) <= num_seqs:
        return msa
    array = np.array([list(seq) for _, seq in msa], dtype=np.bytes_).view(np.uint8)
    optfunc = np.argmax if mode == "max" else np.argmin
    all_indices = np.arange(len(msa))
    indices = [0]
    pairwise_distances = np.zeros((0, len(msa)))
    for _ in range(num_seqs - 1):
        dist = cdist(array[indices[-1:]], array, "hamming")
        pairwise_distances = np.concatenate([pairwise_distances, dist])
        shifted_distance = np.delete(pairwise_distances, indices, axis=1).mean(0)
        shifted_index = optfunc(shifted_distance)
        index = np.delete(all_indices, indices)[shifted_index]
        indices.append(index)
    indices = sorted(indices)
    return [msa[idx] for idx in indices]


def random_select(msa: List[Tuple[str, str]], num_seqs: int) -> List[Tuple[str, str]]:
    import random
    if len(msa) <= num_seqs:
        return msa

    random_indices = random.sample(range(len(msa)), num_seqs)
    return [msa[idx] for idx in random_indices]


def collate_fn_template(data, batch_converter) -> dict:
    x_all = []
    y_all = []
    pdbids = []
    
    for item in data:
        if item != None:
            x = item["msa"]
            y = item["label"]
            if x != (None, None):
                x_all.append(x)
                y_all.append(y)
                pdbids.append(item["pdbid"])

    # Convert to tokens
    try:
        _, _, tokens = batch_converter(x_all)
    except Exception as e:
        print('[LOG] Got exception for: ',pdbids)            
        traceback.print_exception(type(e), e, e.__traceback__) 
        return None

    if tokens.shape[1] > 1024:
        tokens = torch.narrow(tokens,1,0,1024)

    y = torch.from_numpy(np.row_stack(y_all))
    result = {
        "x": tokens,
        "y": y,
        "pdbids": pdbids
    }
    return result


def collate_two_class_fn_template(data, batch_converter) -> dict:
    x_all = []
    y_all_coarse = []
    y_all_fine = []
    pdbids = []    
    
    for item in data:
        x = item["msa"]
        y = item["coarse_y"]
        x_all.append(x)
        y_all_coarse.append(y)
        y = item["fine_y"]
        y_all_fine.append(y)
        pdbids.append(item["pdbid"])        

    # Convert to tokens
    try:
        _, _, tokens = batch_converter(x_all)
    except Exception as e:    
        print('[LOG] Got exception for: ',pdbids)            
        traceback.print_exception(type(e), e, e.__traceback__)
        return None

    if tokens.shape[2] > 1024:
        print('Big MSA',tokens.shape)            
        tokens = torch.narrow(tokens,2,0,1024)

    y_coarse = torch.from_numpy(np.row_stack(y_all_coarse))
    y_fine = torch.from_numpy(np.row_stack(y_all_fine))
    result = {
        "x": tokens,
        "coarse_y": y_coarse,
        "fine_y": y_fine,
        "pdbids": pdbids
    }
    return result


class SimpleHomomerDataLoader(LightningDataModule):
    def __init__(self, params, collater):
        super().__init__()

        self.batch_size = params.bs
        self.params = params
        self.batch_converter = collater
        self.collate_fn = partial(collate_fn_template,
            batch_converter=self.batch_converter
        )
        self.train_pdb_dict, self.valid_pdb_dict, self.test_pdb_dict = get_train_valid_set_seq_split(self.params)  
        self.n_train = len(self.train_pdb_dict)
        self.n_valid = len(self.valid_pdb_dict)
        self.n_test = len(self.test_pdb_dict)

    def get_dataset_size(self):
        return self.n_train+self.n_valid

    def train_dataloader(self):
        train_dataset = SimpleDataset(self.train_pdb_dict, fasta_loader, self.params, 'train')
        if self.params.weighted_sampler == 1:
            train_sampler = DistributedWeightedMySampler(train_dataset, shuffle=True, ddp=False) 
            dl = DataLoader(train_dataset, batch_size=self.batch_size, sampler=train_sampler,
                          collate_fn=self.collate_fn, num_workers=8, pin_memory=True)
        else:
            dl = DataLoader(train_dataset, batch_size=self.batch_size,
                          collate_fn=self.collate_fn, num_workers=8, pin_memory=True)
        return dl


    def test_dataloader(self):
        test_dataset = SimpleDataset(self.test_pdb_dict, fasta_loader, self.params, 'test')    
        return DataLoader(test_dataset, batch_size=self.batch_size,
                          collate_fn=self.collate_fn, num_workers=8, pin_memory=True)

    def val_dataloader(self):
        valid_dataset = SimpleDataset(self.valid_pdb_dict, fasta_loader, self.params, 'valid')
        return DataLoader(valid_dataset, batch_size=self.batch_size, 
                          collate_fn=self.collate_fn, num_workers=8, pin_memory=True)


class CoarseNFineDataLoader(LightningDataModule):
    def __init__(self, params, collater):
        super().__init__()

        self.batch_size = params.bs
        self.params = params
        self.batch_converter = collater
        self.collate_fn = partial(collate_two_class_fn_template,
            batch_converter=self.batch_converter
        )

        self.train_pdb_dict, self.valid_pdb_dict, self.test_pdb_dict = get_train_valid_set_seq_split(self.params)  
        self.n_train = len(self.train_pdb_dict)
        self.n_valid = len(self.valid_pdb_dict)
        self.n_test = len(self.test_pdb_dict)


    # make assignments here (val/train/test split)  called on every process in DDP            
    def get_dataset_size(self):
        return self.n_train+self.n_valid


    def train_dataloader(self):
        train_dataset = CoarseNFineDataset(self.train_pdb_dict, fasta_loader, self.params, 'train')
        if self.params.weighted_sampler == 1:
            print('Setting up weighted sampler for training data..')
            train_sampler = DistributedWeightedMySampler(train_dataset, num_example_per_epoch=self.params.limit, shuffle=True, ddp=False)
            dl = DataLoader(train_dataset, batch_size=self.batch_size, sampler=train_sampler,
                          collate_fn=self.collate_fn, num_workers=8, pin_memory=True)
        else:
            print('Using unweighted sampler for training data..')
            dl = DataLoader(train_dataset, batch_size=self.batch_size, 
                          collate_fn=self.collate_fn, num_workers=8, pin_memory=True)         
        return dl 

    
    def test_dataloader(self):
        test_dataset = CoarseNFineDataset(self.test_pdb_dict, fasta_loader, self.params, 'test')    
        return DataLoader(test_dataset, batch_size=1, 
                          collate_fn=self.collate_fn, num_workers=4, pin_memory=True)

    def val_dataloader(self):
        valid_dataset = CoarseNFineDataset(self.valid_pdb_dict, fasta_loader, self.params, 'valid')
        return DataLoader(valid_dataset, batch_size=1, 
                          collate_fn=self.collate_fn, num_workers=4, pin_memory=True)



class CoarseNFineDataset(data.Dataset):
    def __init__(self, datadict, fasta_loader, params, dataset_name):
        self.pdb_dict = datadict
        self.pdbids = list(datadict.keys()) 
        ## this sets the ordering of the examples. At index 0 is the protein whose PDB-id is the first key
        self.fasta_loader = fasta_loader
        self.params = params
        self.dataset_name = dataset_name
        ## construct cluster mapping
        self.cluster_labels = []
        if self.params.weighted_sampler == 1:
            for pdbid in self.pdbids:  # val is a tuple: ('PDBID', 'HASH', 'CLUSTER', 'SYMM', 'LENGTH')
                val = self.pdb_dict[pdbid]
                self.cluster_labels.append(val[2])

        self.rank = self.__get_rank__()
        
    def __get_rank__(self):
        if not dist.is_available():
            raise RuntimeError("Requires distributed package to be available")
        return dist.get_rank()
        

    def __len__(self):
        return len(self.pdbids)


    def get_examples_weights(self):
        cluster_sizes = {}      # key: clus, val: number of proteins
        sum_chain_lengths = {}  # key: clus, val: sum of chain length
        for key, val in self.pdb_dict.items():  # val is a tuple: ('PDBID', 'HASH', 'CLUSTER', 'SYMM', 'LENGTH')
            if val[2] not in cluster_sizes:
                cluster_sizes[val[2]] = 0
                sum_chain_lengths[val[2]] = 0
            sum_chain_lengths[val[2]] += val[4]
            cluster_sizes[val[2]] += 1

        # create weights to be in order of pdbids as that's the order in which examples are indexed in the Dataset
        # inversely weighted based on cluster size
        weights = []
        for _pdbid in self.pdbids:
            weights.append(1/cluster_sizes[self.pdb_dict[_pdbid][2]])

        return weights
    

    def map_labels_to_vectors(self, label, label_type):
        if label_type=="coarse":
            symm_map = symm_to_coarselabel_map
            nclasses = self.params.n_classes1
        elif label_type=="fine":
            symm_map = symm_to_label_map
            nclasses = self.params.n_classes2

        homomer_label_vecs = []
        # map labels to integers and one-hot encode them
        for l in label.split(' '):
            if l in symm_map:
                homomer_label = symm_map[l]
            else:
                homomer_label = nclasses - 1
            homomer_label_vecs.append(one_hot_encode(homomer_label, num_classes=nclasses))
        # address multilabel cases
        if len(homomer_label_vecs) > 1:
            homomer_label_vecs = np.row_stack(homomer_label_vecs)
            homomer_label_vecs = np.sum(homomer_label_vecs,axis=0)
            homomer_label_vecs = (homomer_label_vecs>0).astype(float)
        else:
            homomer_label_vecs = homomer_label_vecs[0].astype(float)

        return homomer_label_vecs


    def __getitem__(self, index):
        ID = self.pdb_dict[self.pdbids[index]]  # ID is a tuple: ('PDBID', 'HASH', 'CLUSTER', 'SYMM')
        try:
            msa = self.fasta_loader((ID[0], str(ID[1])), self.params)
            label = ID[3]
            ## get multilabel fine label vector
            homomer_label_vecs_coarse = self.map_labels_to_vectors(label, label_type="coarse")
            homomer_label_vecs_fine = self.map_labels_to_vectors(label, label_type="fine")
            data = {"msa" : msa,
                "coarse_y": homomer_label_vecs_coarse, "fine_y": homomer_label_vecs_fine, "pdbid": ID[0]}            

        except Exception as e:
            print('[LOG] Got exception for: ',index,self.pdbids[index],e)            
            traceback.print_exception(type(e), e, e.__traceback__)          
            index = np.random.randint(self.__len__) 
            data = self.__getitem__(index)      
            
        return data


class TestSequencesLoader(LightningDataModule):
    def __init__(self, params, collater):
        super().__init__()

        self.batch_converter = collater
        self.collate_fn = partial(collate_fn_template,
            batch_converter=self.batch_converter
        )

        self.params = params
        self.params.test_mode = True
    
    def train_dataloader(self):
        print('Not a training dataloader')
        return None            

    def val_dataloader(self):
        print('Not a validation dataloader')
        return None                    

    def test_dataloader(self):
        test_dataset = LazyFileloaderDataset(self.params)
        return DataLoader(test_dataset, batch_size=self.params.bs, shuffle=False,
                          collate_fn=self.collate_fn, num_workers=4, pin_memory=True)


class TestFASTALoader(LightningDataModule):
    def __init__(self, params, collater):
        super().__init__()

        self.batch_converter = collater
        self.collate_fn = partial(collate_fn_template,
            batch_converter=self.batch_converter
        )        

        self.params = params
        self.params.test_mode = True
        label_dict = {}
        if params.meta_data_file:
            print('Reading: ',self.params.meta_data_file)
            df = pd.read_csv(self.params.meta_data_file,header=0,sep=',',dtype='str')
            for _, row in df.iterrows():
                label_dict[row['pdbid']] = row['symm']

        self.test_pdb_dict = {}  
        # dict of tuples ('PDBID', 'HASH', 'CLUSTER', 'SYMM', 'LENGTH')
        for root, _, files in os.walk(self.params.data_dir):
            for fname in files:
                filepath = os.path.join(root, fname)
                pdbid = filepath.split('/')[-1].split('.')[0]
                if pdbid in label_dict:
                    self.test_pdb_dict[pdbid] = (pdbid, filepath, '###', label_dict[pdbid], 0)
                else:
                    if len(label_dict) == 0:  
                        self.test_pdb_dict[pdbid] = (pdbid, filepath, '###', 0, 0)
    
    def train_dataloader(self):
        print('Not a training dataloader')
        return None            

    def val_dataloader(self):
        print('Not a validation dataloader')
        return None                    

    def test_dataloader(self):
        test_dataset = CoarseNFineJointDataset(self.test_pdb_dict, fasta_loader, self.params, dataset_name='test')            
        return DataLoader(test_dataset, batch_size=self.params.bs, 
                          collate_fn=self.collate_fn, num_workers=8, pin_memory=True)



class CoarseNFineJointDataLoader(LightningDataModule):

    def __init__(self, params, collater):
        super().__init__()

        self.batch_size = params.bs
        self.params = params

        self.batch_converter = collater
        self.collate_fn = partial(collate_fn_template,
            batch_converter=self.batch_converter
        )

        self.train_pdb_dict, self.valid_pdb_dict, self.test_pdb_dict = get_train_valid_set_seq_split(self.params)  
        self.n_train = len(self.train_pdb_dict)
        self.n_valid = len(self.valid_pdb_dict)
        self.n_test = len(self.test_pdb_dict)

    # make assignments here (val/train/test split), called on every process in DDP            
    def get_dataset_size(self):
        return self.n_train+self.n_valid

    def train_dataloader(self):
        train_dataset = CoarseNFineJointDataset(self.train_pdb_dict, fasta_loader, self.params, dataset_name='train')        
        if self.params.weighted_sampler == 1:
            print('Setting up weighted sampler for training data..')
            train_sampler = DistributedWeightedMySampler(train_dataset, num_example_per_epoch=self.params.limit, shuffle=True, ddp=False)            
            dl = DataLoader(train_dataset, batch_size=self.batch_size, sampler=train_sampler,
                          collate_fn=self.collate_fn, num_workers=8, pin_memory=True)
        else:
            print('Using unweighted sampler for training data..')
            dl = DataLoader(train_dataset, batch_size=self.batch_size, 
                          collate_fn=self.collate_fn, num_workers=8, pin_memory=True)
        return dl 
    
    def test_dataloader(self):
        test_dataset = CoarseNFineJointDataset(self.test_pdb_dict, fasta_loader, self.params, dataset_name='test')    
        return DataLoader(test_dataset, batch_size=self.batch_size, 
                          collate_fn=self.collate_fn, num_workers=8, pin_memory=True)

    def val_dataloader(self):
        valid_dataset = CoarseNFineJointDataset(self.valid_pdb_dict, fasta_loader, self.params, dataset_name='valid')
        return DataLoader(valid_dataset, batch_size=self.batch_size,
                          collate_fn=self.collate_fn, num_workers=8, pin_memory=True)


class CoarseNFineJointDataset(data.Dataset):
    def __init__(self, datadict, loader, params, dataset_name):
        self.pdb_dict = datadict
        self.pdbids = list(datadict.keys()) ## this sets the ordering of the examples. At index 0 is the protein whose PDB-id is the first key
        self.loader = loader
        self.params = params
        self.dataset_name = dataset_name
        ## construct cluster mapping
        self.cluster_labels = []
        if self.params.weighted_sampler == 1:
            print('[LOG] using weighted sampler')            
            for pdbid in self.pdbids:  # val is a tuple: ('PDBID', 'HASH', 'CLUSTER', 'SYMM', 'LENGTH')
                val = self.pdb_dict[pdbid]
                self.cluster_labels.append(val[2])

        if self.params.use_soft_labels:
            print('[LOG] using soft labels')                

    def __get_rank__(self):
        if not dist.is_available():
            raise RuntimeError("Requires distributed package to be available")
        return dist.get_rank()            

    def __len__(self):
        return len(self.pdbids)

    def get_cluster_labels(self):
        return self.cluster_labels

    def get_examples_weights(self):
        cluster_sizes = {}      # key: clus, val: number of proteins
        for key, val in self.pdb_dict.items():  # val is a tuple: ('PDBID', 'HASH', 'CLUSTER', 'SYMM', 'LENGTH')
            if val[2] not in cluster_sizes:
                cluster_sizes[val[2]] = 0
            cluster_sizes[val[2]] += 1

        # create weights to be in order of pdbids as that's the order in which examples are indexed in the Dataset
        # weight = inversely weighted based on cluster size
        weights = []
        for _pdbid in self.pdbids:
            weights.append(1/cluster_sizes[self.pdb_dict[_pdbid][2]])

        return weights
        


    def get_exp_decay_wts(self, num: int) -> list:
        if num==1:
            return [1.0]
        wts = []
        starter = 1.0
        denom = 1.1**num
        powers = [1,6,1,0.5]
        for i in range(num):
            starter = starter/(denom**powers[i])
            wts.append(starter)
        return wts

    # map to soft labels
    def map_labels_to_soft_vectors(self, label):
        symm_map = symm_to_joint_label_map
        nclasses = self.params.n_classes
        homomer_label_vecs = []
        # map labels to integers and one-hot encode them add soft labels rather than 1, in the order the labels are processed
        num_labels = len(label.split(' '))
        soft_labels = self.get_exp_decay_wts(num_labels)
        labels = label.split(' ')
        for lidx in range(num_labels):
            if labels[lidx] in symm_map:
                homomer_label = symm_map[labels[lidx]]
                if type(homomer_label) is list:
                    for hl in homomer_label:
                        homomer_label_vecs.append(soft_labels[lidx] * one_hot_encode(hl, num_classes=nclasses))
                else:
                    homomer_label_vecs.append(soft_labels[lidx] * one_hot_encode(homomer_label, num_classes=nclasses))
            else:
                homomer_label = nclasses - 1
                homomer_label_vecs.append(1e-5 * one_hot_encode(homomer_label, num_classes=nclasses))
        # address multilabel cases
        if len(homomer_label_vecs) > 1:
            homomer_label_vecs = np.row_stack(homomer_label_vecs)
            homomer_label_vecs = np.sum(homomer_label_vecs,axis=0)
        else:
            homomer_label_vecs = homomer_label_vecs[0].astype(float)

        return homomer_label_vecs

    ## one-hot encodes labels
    def map_labels_to_vectors(self, label):
        symm_map = symm_to_joint_label_map
        nclasses = self.params.n_classes
        homomer_label_vecs = []
        # map labels to integers and one-hot encode them
        for l in label.split(' '):
            if l not in symm_map:
                if l.startswith('C'):
                    l = 'C17'
                if l.startswith('D'):
                    l = 'D12'                    
            if l in symm_map:
                homomer_label = symm_map[l]
                if type(homomer_label) is list:
                    for hl in homomer_label:
                        homomer_label_vecs.append(one_hot_encode(hl, num_classes=nclasses))
                else:
                    homomer_label_vecs.append(one_hot_encode(homomer_label, num_classes=nclasses))
            else:
                homomer_label = nclasses - 1
                homomer_label_vecs.append(one_hot_encode(homomer_label, num_classes=nclasses))
        # address multilabel cases
        if len(homomer_label_vecs) > 1:
            homomer_label_vecs = np.row_stack(homomer_label_vecs)
            homomer_label_vecs = np.sum(homomer_label_vecs,axis=0)
            homomer_label_vecs = (homomer_label_vecs>0).astype(float)
        else:
            homomer_label_vecs = homomer_label_vecs[0].astype(float)
        return homomer_label_vecs


    def __getitem__(self, index):
        ID = self.pdb_dict[self.pdbids[index]]  # ID is a tuple: ('PDBID', 'HASH', 'CLUSTER', 'SYMM')
        try:
            fasta = self.loader((ID[0], str(ID[1])), self.params)
            label = ID[3] ## string: "C1, C2, C3"
            ## get multilabel label vector
            if self.params.use_soft_labels:
                homomer_label_vecs_joint = self.map_labels_to_soft_vectors(label)
            else:
                homomer_label_vecs_joint = self.map_labels_to_vectors(label)                
            data = {"msa" : fasta,
                "label": homomer_label_vecs_joint, "pdbid": ID[0]}

        except Exception as e:
            print('[LOG] Got exception for: ',index,self.pdbids[index],e)            
            traceback.print_exception(type(e), e, e.__traceback__)          
            index = index - 1 if index > 0 else index + 1 
            data = self.__getitem__(index)      
            
        return data




class SimpleDataset(Dataset):
    def __init__(self, pdb_dict, fasta_loader, params, dataset_name):
        self.pdb_dict = pdb_dict
        self.pdb_IDs = list(pdb_dict.keys()) ## this sets the ordering of the examples. At index 0 is the protein whose PDB-id is the first key
        self.fasta_loader = fasta_loader
        self.params = params
        self.dataset_name = dataset_name
        ## construct cluster mapping
        self.cluster_labels = []
        if self.params.weighted_sampler == 1:
            for pdbid in self.pdb_IDs:  # val is a tuple: ('PDBID', 'HASH', 'CLUSTER', 'SYMM', 'LENGTH')
                val = self.pdb_dict[pdbid]
                self.cluster_labels.append(val[2])

        self.rank = self.__get_rank__()

    def __get_rank__(self):
        if not dist.is_available():
            raise RuntimeError("Requires distributed package to be available")
        #return dist.get_rank()


    def get_examples_weights(self):
        cluster_sizes = {}      # key: clus, val: number of proteins
        for key, val in self.pdb_dict.items():  # val is a tuple: ('PDBID', 'HASH', 'CLUSTER', 'SYMM', 'LENGTH')
            if val[2] not in cluster_sizes:
                cluster_sizes[val[2]] = 0
            cluster_sizes[val[2]] += 1

        # create weights to be in order of pdbids as that's the order in which examples are indexed in the Dataset
        # weight = inversely weighted based on cluster size
        weights = []
        for _pdbid in self.pdb_IDs:
            weights.append(1/cluster_sizes[self.pdb_dict[_pdbid][2]])

        return weights


    def __len__(self):
        return len(self.pdb_IDs)

    def map_labels_to_vectors(self, label, label_type):
        if label_type=="coarse":
            symm_map = symm_to_coarselabel_map
            nclasses = len(np.unique(list(symm_to_coarselabel_map.values())))
        elif label_type=="fine":
            symm_map = symm_to_label_map
            nclasses = len(np.unique(list(symm_to_label_map.values())))

        homomer_label_vecs = []
        # map labels to integers and one-hot encode them
        for l in label.split(' '):
            if l in symm_map:
                homomer_label = symm_map[l]
            else:
                homomer_label = 0 
            homomer_label_vecs.append(one_hot_encode(homomer_label, num_classes=nclasses))
        # address multilabel cases
        if len(homomer_label_vecs) > 1:
            homomer_label_vecs = np.row_stack(homomer_label_vecs)
            homomer_label_vecs = np.sum(homomer_label_vecs,axis=0)
            homomer_label_vecs = (homomer_label_vecs>0).astype(float)
        else:
            homomer_label_vecs = homomer_label_vecs[0].astype(float)

        return homomer_label_vecs


    def __getitem__(self, index):
        ID = self.pdb_dict[self.pdb_IDs[index]]  # ID is a tuple: ('PDBID', 'HASH', 'CLUSTER', 'SYMM')
        try:
            msa = self.fasta_loader((ID[0], str(ID[1])), self.params)
            label = ID[3]
            ## get multilabel fine label vector
            homomer_label_vecs = self.map_labels_to_vectors(label, label_type="fine")
            data = {"msa" : msa,
                "label": homomer_label_vecs, "pdbid": ID[0]}

        except Exception as e:
            print('[LOG] Got exception for: ',index,self.pdb_IDs[index],e)            
            traceback.print_exception(type(e), e, e.__traceback__)          
            index = np.random.randint(self.__len__()) 
            data = self.__getitem__(index)      
            
        return data


class LazyFileloaderDataset(Dataset):
    def __init__(self, params):
        self.params = params
        print('LazyFileLoader Reading: ',self.params.meta_data_file)
        df = pd.read_csv(self.params.meta_data_file,header=0,sep=',',dtype='str')
        self.header = df.columns
        self.length = len(df.index)
        print('....with header:',self.header,' and length:',self.length)
        del df

    def __len__(self):
        return self.length        

    ## one-hot encodes labels
    def map_labels_to_vectors(self, label):
        symm_map = symm_to_joint_label_map
        nclasses = self.params.n_classes
        homomer_label_vecs = []
        # map labels to integers and one-hot encode them
        for l in label.split(' '):
            if l not in symm_map:
                if l.startswith('C'):
                    l = 'C17'
                if l.startswith('D'):
                    l = 'D12'                    
            if l in symm_map:
                homomer_label = symm_map[l]
                if type(homomer_label) is list:
                    for hl in homomer_label:
                        homomer_label_vecs.append(one_hot_encode(hl, num_classes=nclasses))
                else:
                    homomer_label_vecs.append(one_hot_encode(homomer_label, num_classes=nclasses))
            else:
                homomer_label = nclasses - 1
                homomer_label_vecs.append(one_hot_encode(homomer_label, num_classes=nclasses))
        # address multilabel cases
        if len(homomer_label_vecs) > 1:
            homomer_label_vecs = np.row_stack(homomer_label_vecs)
            homomer_label_vecs = np.sum(homomer_label_vecs,axis=0)
            homomer_label_vecs = (homomer_label_vecs>0).astype(float)
        else:
            homomer_label_vecs = homomer_label_vecs[0].astype(float)
        return homomer_label_vecs



    def __getitem__(self, index):
        try:
            if index==0:  ## since this is the header line
                index = index+1
            datarow = linecache.getline(self.params.meta_data_file, index).rstrip().split(',')
            seqidx = np.where(self.header == 'SEQUENCE')[0][0]
            pdbidx = np.where(self.header == 'PDBID')[0][0]
            if 'SYMM' in self.header:
                symm = datarow[np.where(self.header == 'SYMM')[0][0]]
            else:
                symm = 'C1'

            symm_vec = self.map_labels_to_vectors(symm)
            data = {"msa" : (datarow[pdbidx], datarow[seqidx]), "pdbid": datarow[pdbidx], "label": symm_vec}
        except Exception as e:
            print('[LOG] Got exception for: ',index)
            traceback.print_exception(type(e), e, e.__traceback__)
            index = index - 1 if index > 0 else index + 1
            data = self.__getitem__(index)
        return data


class WeightedHomomerDataLoader(LightningDataModule):

    def __init__(self, params, collater):
        super().__init__()

        self.batch_size = params.bs
        self.params = params
        self.batch_converter = collater
        self.collate_fn = partial(collate_fn_template,
            batch_converter=self.batch_converter
        )

        datadict = read_data(params)
        self.train_pdb_dict = datadict['train_split']
        self.valid_pdb_dict = datadict['valid_split']
        self.test_pdb_dict = datadict['test_split']
        self.clusterid_to_pdbids = datadict['clusterid_to_pdbids']
        self.n_train = len(self.train_pdb_dict)
        self.n_valid = len(self.valid_pdb_dict)
        self.n_test = len(self.test_pdb_dict)


    def get_dataset_size(self):
        return self.n_train+self.n_valid

    def train_dataloader(self):
        train_dataset = WeightedDataset(self.train_pdb_dict, fasta_loader, self.params)
        train_sampler = DistributedWeightedMySampler(train_dataset, num_example_per_epoch=self.params.num_example_per_epoch, shuffle=True, ddp=False)        
        return DataLoader(train_dataset, batch_size=self.batch_size, sampler=train_sampler,
                          collate_fn=self.collate_fn, num_workers=4, pin_memory=True)
    
    def test_dataloader(self):
        test_dataset = SimpleDataset(self.test_pdb_dict, fasta_loader, self.params)    
        return DataLoader(test_dataset, batch_size=self.batch_size,
                          collate_fn=self.collate_fn, num_workers=4, pin_memory=True)

    def val_dataloader(self):
        valid_dataset = SimpleDataset(self.valid_pdb_dict, fasta_loader, self.params)
        return DataLoader(valid_dataset, batch_size=self.batch_size, 
                          collate_fn=self.collate_fn, num_workers=4, pin_memory=True)


class WeightedDataset(Dataset):

    def __init__(self, pdb_dict, pdb_loader, params):
        self.pdb_dict = pdb_dict              ## each entry a tuple: ('PDBID', 'HASH', 'CLUSTER', 'SYMM', 'LENGTH')
        self.pdb_IDs = list(pdb_dict.keys())  ## this sets the ordering of the examples. At index 0 is the protein whose PDB-id is the first key
        self.pdb_loader = pdb_loader
        self.params = params

        # create weights to be in order of pdbids as that's the order in which examples are indexed in the WeightedDataset
        cluster_sizes = {}      # key: clus, val: number of proteins
        sum_chain_lengths = {}  # key: clus, val: sum of chain length
        for key, val in pdb_dict.items():  # val is a tuple: ('PDBID', 'HASH', 'CLUSTER', 'SYMM', 'LENGTH')
            if val[2] not in cluster_sizes:
                cluster_sizes[val[2]] = 0
                sum_chain_lengths[val[2]] = 0
            sum_chain_lengths[val[2]] += val[4]
            cluster_sizes[val[2]] += 1

        weights = []
        if params.weighting_scheme == 'cluster_size':
            for _pdbid in self.pdb_IDs:  ## weight based on cluster size
                weights.append(1/cluster_sizes[pdb_dict[_pdbid][2]])
        elif params.weighting_scheme == 'none':
            weights = [1.0] * len(self.pdb_IDs)
        else:
            # weight based on avg chain length
            for _pdbid in self.pdb_IDs:
                clusterid = pdb_dict[_pdbid][2]
                w = (1/512.)*max(min(float(sum_chain_lengths[clusterid] / cluster_sizes[clusterid]),512.),256.)
                weights.append(w)

        print('Max weight: ',max(weights),' Min weight:',min(weights))
        np.savetxt(fname='/tmp/weights.txt',X=weights,delimiter='\n')

        self.weights = weights


    def get_example_weights(self):
        return self.weights
    
    
    def __len__(self):
        return len(self.pdb_IDs)

    def __getitem__(self, index):
        ID = self.pdb_dict[self.pdb_IDs[index]]  # ID is a tuple: ('PDBID', 'HASH', 'CLUSTER', 'SYMM', 'LENGTH')
        try:
            data = self.pdb_loader((ID[0], str(ID[1]), str(ID[2]), ID[3], self.pdb_IDs[index]), self.params)  # data is a dict {"msa", "label"}
        except Exception as e:
            print('[LOG] Got exception for: ',index,self.pdb_IDs[index],e)            
            traceback.print_exception(type(e), e, e.__traceback__)          
            index = index - 1 if index > 0 else index + 1 
            data = self.__getitem__(index)      
        return data



class ClusterBasedSampler(data.Sampler):
    def __init__(self, dataset, num_example_per_epoch=100, \
                 num_replicas=None, rank=None, replacement=False, shuffle=False):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()

        assert num_example_per_epoch % num_replicas == 0 
        self.num_example_per_epoch = num_example_per_epoch

        self.dataset = dataset
        self.batch_size = dataset.params.bs        
        self.num_replicas = num_replicas
        self.total_size = len(dataset)
        self.rank = rank
        self.epoch = 0
        self.replacement = replacement
        self.shuffle = shuffle

        self.cluster_labels = dataset.get_cluster_labels()
        self.cluster_indices_dict = self._create_cluster_dict()

        if self.total_size < num_example_per_epoch:
            self.num_example_per_epoch = self.total_size
        self.num_samples = self.num_example_per_epoch // self.num_replicas            
        print('Num_examples_per_epoch, world_size, total_size, num_samples_per_gpu: ',
              self.num_example_per_epoch, num_replicas, self.total_size, self.num_samples)       

    def _create_cluster_dict(self):
        cluster_indices_dict = {}
        for idx, cluster in enumerate(self.cluster_labels):
            if cluster in cluster_indices_dict:
                cluster_indices_dict[cluster].append(idx)
            else:
                cluster_indices_dict[cluster] = [idx]
        return cluster_indices_dict

    def __iter__(self):
        batches = []
        for proteins in self.cluster_indices_dict.values():
            proteins = torch.tensor(proteins)
            if self.shuffle:
                proteins = proteins[torch.randperm(len(proteins))]
            batches = batches + [proteins[i:i+self.batch_size] for i in range(0, len(proteins), self.batch_size)]
        
        batches = torch.tensor(batches)
        if self.shuffle:
            batches = batches[torch.randperm(len(batches))]

        indices = batches[0:self.num_example_per_epoch]
        indices = indices[self.rank:len(indices):self.num_replicas]
        print('indices generated per gpu:',len(indices),'num_samples expected per gpu:',self.num_samples)
        assert len(indices) == self.num_samples

        return iter(indices.tolist())

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch



class DistributedWeightedMySampler(data.Sampler):
    def __init__(self, dataset, num_example_per_epoch=100, \
                 num_replicas=None, rank=None, replacement=False, shuffle=False, ddp=False):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()


        self.num_example_per_epoch=int(len(dataset))
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.total_size = len(dataset)
        self.rank = rank
        self.epoch = 0
        self.ddp = ddp
        self.replacement = replacement
        self.weights = dataset.get_examples_weights()
        ## normalize weights
        allsum = sum(self.weights)
        print('weighting before:',allsum,self.weights[0:10])
        if(allsum != 1):
            for w in range(len(self.weights)):
                self.weights[w] = self.weights[w] / allsum
        print('weighting after:',sum(self.weights),self.weights[0:10])
        assert round(sum(self.weights)) == 1

        self.shuffle = shuffle

        if self.total_size < self.num_example_per_epoch:
            self.num_example_per_epoch = self.total_size
        self.num_samples = self.num_example_per_epoch // self.num_replicas            
        print('Num_examples_per_epoch, world_size, total_size, num_samples_per_gpu: ',
              self.num_example_per_epoch, num_replicas, self.total_size, self.num_samples)       


    def __iter__(self):
        # get indices
        indices = torch.arange(len(self.dataset))
        g = torch.Generator()
        g.manual_seed(self.epoch)
        # set sampling based on weight of example
        indices = indices[torch.multinomial(torch.tensor(self.weights), self.num_example_per_epoch, self.replacement, generator=g)]
        if self.shuffle:
            indices = indices[torch.randperm(len(indices))]
        if self.ddp:
            # do manual split per each gpu
            indices = indices[self.rank:len(indices):self.num_replicas]
            assert len(indices) == self.num_samples

        return iter(indices.tolist())

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch
