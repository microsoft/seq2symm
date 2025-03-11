# Introduction 
Source code for Seq2Symm: Rapid and accurate prediction of protein homo-oligomer symmetry, published here:

Meghana Kshirsagar, Artur Meller, Ian R. Humphreys, Samuel Sledzieski, Yixi Xu, Rahul Dodhia, Eric Horvitz, Bonnie Berger, Gregory R. Bowman, Juan Lavista Ferres, David Baker, Minkyung Baek, _Nature Communications_

https://www.nature.com/articles/s41467-025-57148-3

Seq2Symm takes a single sequence as input

# Getting Started

There are two options: do inference using Google Colab or a local installation where you can do both inference or training

## Google Colab notebook for small-scale inference

This self-contained notebook will do inference on input protein sequences

https://colab.research.google.com/drive/1ptQTyC22ExxJ3BnSK6dnPaCeg7J8i3le#scrollTo=CJm12VgOiDfj

## Conda environment for local installation and large-scale inference and training

Dependencies are in the yaml file esm2_finetune.yaml

```
conda env create --name esm2 --file=esm2_finetune.yaml
```

# Downloads

  1.  the predictions from the model on various datasets, predictions on proteomes
      http://files.ipd.uw.edu/pub/seq2symm/predictions.zip
  
  2.  the trained model
      http://files.ipd.uw.edu/pub/seq2symm/ESM2_model.ckpt

  3.  data download links are here
      https://github.com/microsoft/seq2symm/tree/main/datasets

      All code, datasets, model, predictions are also available on Zenodo: http://doi.org/10.5281/zenodo.14659968
      

# Training the model

```
python src/finetune.py --meta_data_file ../datasets/homomer_pdbids_hash_clusterid_labels_sampled.csv --data_dir ../datasets/ --model_dir models/ --output_model seq2symm --output_dir outputs/seq2symm --bs 16 --data_splits_file ../datasets/train_val_test_splits.pkl --limit 65536 --granularity 3 --n_classes 20 --n_epoch 100 --weighted_sampler 1
```

# Predicting using the model

## Jupyternotebook
A jupyter notebook is available at src/load_chkpt_and_predict.ipynb that shows examples of how this is done for two different file formats

## Predicting via command line
This script is available courtesy of <a href="https://github.com/MoritzErtelt">Moritz Ertlet</a>: 

The  `src/predict_oligmerization.py` script predicts the oligomerization states of protein sequences provided in a FASTA file, outputting the probabilities for each symmetry state with a probability >= 1%.
```
python predict_oligomerization.py -input_file <input_fasta> -chkpt_file <model_checkpoint> [-output_file <results.csv>] [-batch_size <n>]
```
* `-input_file`: Path to the input FASTA file containing the sequences for prediction (required).
* `-chkpt_file`: Path to the model checkpoint file (required).
* `-output_file`: Path to save the prediction results as a CSV file (default: ./results.csv).
* `-batch_size`: Batch size for processing sequences (default: 1).

Example output:
```
fasta_id, predicted_oligomerization_state
ID1, "{'C1': 0.9, 'C2': 0.1, 'Other': 0.001}"
ID2, "{'C4': 0.85, 'C6': 0.15, 'D3': 0.01}"
```
Which you can load with pandas like this:
```
import pandas as pd
import ast
df = pd.read_csv("results.csv")
df['predicted_oligomerization_state'] = df['predicted_oligomerization_state'].apply(ast.literal_eval)
```

