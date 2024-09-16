import argparse
import torch
import esm
import csv
from data_loader import TestFASTALoader
from lightning.pytorch import Trainer
from finetune import ESMFinetuner
import numpy as np

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

# Function to parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Predict oligomerization states from FASTA sequences.")
    parser.add_argument('-input_file', type=str, required=True, help='Input FASTA file with sequences')
    parser.add_argument('-output_file', type=str, default="./results.csv", help='Output CSV file to save predictions')
    parser.add_argument('-chkpt_file', type=str, required=True, help='Path to the model checkpoint file')
    parser.add_argument('-batch_size', type=int, default=1, help='Batch size for processing sequences (default: 1)')
    return parser.parse_args()

# Function to run predictions
def run_predictions(input_file, output_file, chkpt_file, batch_size):
    # Define model parameters
    class Params:
        def __init__(self):
            self.test_mode = True
            self.n_classes = 17
            self.granularity = 2
            self.bs = batch_size
            self.output_dir = "./outputs"
            self.suffix = "seq2symm"
            self.n_epoch = 1
            self.num_layers_frozen = 31
            self.fasta_file = input_file
            self.weighted_sampler = False
            self.use_soft_labels = False
            self.chkpt_file = chkpt_file

    # Initialize parameters and model
    params = Params()
    task = ESMFinetuner(params=params)

    # Load the pre-trained model
    task = task.load_from_checkpoint(params.chkpt_file)
    task.eval()  # Set model to evaluation mode

    # Initialize the test dataloader for the input FASTA file
    dataloader = TestFASTALoader(params=params, collater=task.batch_converter)

    # Initialize the Trainer
    trainer = Trainer(accelerator="cuda" if torch.cuda.is_available() else "cpu", max_epochs=params.n_epoch)

    # Run prediction
    predictions = trainer.predict(model=task, dataloaders=[dataloader.test_dataloader()])

    # Save predictions to the output CSV file
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['fasta_id', 'predicted_oligomerization_state'])

        for pred in predictions:
            pred_state = pred[1]
            fasta_id = pred[0][0]
            pred_state = pred_state.squeeze(0)

            # Calculate softmax probabilities for all oligomerization states
            probabilities = torch.softmax(pred_state, dim=0).tolist()

            # Map probabilities to oligomerization states and filter out very small values
            symm_probs = {joint_label_to_symm_map[i]: round(prob, 4) for i, prob in enumerate(probabilities) if prob >= 0.01}

            # Sort the oligomerization states by probability
            sorted_symm_probs = dict(sorted(symm_probs.items(), key=lambda item: item[1], reverse=True))

            # Write to the CSV file
            writer.writerow([fasta_id, str(sorted_symm_probs)])

# Main execution
if __name__ == "__main__":
    args = parse_args()
    run_predictions(args.input_file, args.output_file, args.chkpt_file, args.batch_size)

