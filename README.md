# Introduction 
Source code for 
Seq2Symm: Rapid and accurate prediction of protein homo-oligomer symmetry

Rapid prediction of homo-oligomer symmetries using a single sequence as input

# Getting Started
Dependencies are in the yaml file esm2_finetune.yaml

```
conda env create --name esm2 --file=esm2_finetune.yaml
```

# Downloads

  1.  the predictions from the model on various datasets, predictions on proteomes
      http://files.ipd.uw.edu/pub/seq2symm/predictions.zip
  
  2.  the trained model
      http://files.ipd.uw.edu/pub/seq2symm/ESM2_model.ckpt
      

# Training the model

```
python src/finetune.py --meta_data_file ../datasets/homomer_pdbids_hash_clusterid_labels_sampled.csv --data_dir ../datasets/ --model_dir models/ --output_model seq2symm --output_dir outputs/seq2symm --bs 16 --data_splits_file ../datasets/train_val_test_splits.pkl --limit 65536 --granularity 3 --n_classes 20 --n_epoch 100 --weighted_sampler 1
```
