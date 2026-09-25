# External repository audit

Audited at 2026-09-24T23:07:53.370184+00:00

AmpLyze has no reliably identified official checkpoint and is treated as a methodological reference only. Predictors trained on DBAASP/Hemolytik are marked `potentially_contaminated` relative to our DBAASP-derived locked test.

## HemoPI2

- URL: https://github.com/raghavagps/HemoPI2
- HEAD SHA: `2b67a5c85422b25ae847100ebaa81ad586950928`
- License: GPL-3.0
- Target: quantitative HC50 + binary hemolysis (threshold 100 µM)
- Files in tree: 16
- Checkpoint candidate in repo: False
- Local checkpoint present: True
- Inference reproducible: True
- Potentially contaminated vs our test: True
- Status: ok
- Notes: Vendored under external/hemopi2 with Model/HemoPI2_reg.sav. Trained on DBAASP+Hemolytik → potentially contaminated vs our test.
- Interesting files (first 40):
  - `Dataset/cross_val_dataset.csv`
  - `Dataset/independent_dataset.csv`
  - `Feature_correlation.csv`
  - `example.fasta`

## ConsAMPHemo

- URL: https://github.com/Cpillar/ConsAMPHemo
- HEAD SHA: `950fb333d2126f79dda94a673e85ae1965d50d11`
- License: NOASSERTION
- Target: HC50 regression + classification
- Files in tree: 31
- Checkpoint candidate in repo: True
- Local checkpoint present: False
- Inference reproducible: True
- Potentially contaminated vs our test: True
- Status: ok
- Notes: Siamese/contrastive GRU + ProtBERT + XGBoost. No SPDX license.
- Interesting files (first 40):
  - `Dataset/.DS_Store`
  - `Dataset/S1/.DS_Store`
  - `Dataset/S1/S_1.csv`
  - `Dataset/S1/model/.DS_Store`
  - `Dataset/S1/model/S1.pth.pl`
  - `Dataset/S1/test.csv`
  - `Dataset/S1/train.csv`
  - `Dataset/S2/.DS_Store`
  - `Dataset/S2/S_2.csv`
  - `Dataset/S2/model/.DS_Store`
  - `Dataset/S2/model/S2.pl`
  - `Dataset/S2/test.csv`
  - `Dataset/S2/train.csv`
  - `Dataset/S3/.DS_Store`
  - `Dataset/S3/S_3.csv`
  - `Dataset/S3/model/.DS_Store`
  - `Dataset/S3/model/S3.pth.pl`
  - `Dataset/S3/test.csv`
  - `Dataset/S3/train.csv`
  - `Dataset/regression/.DS_Store`
  - `Dataset/regression/Hemo_regression.csv`
  - `Dataset/regression/model/XGB_model_Hemo.joblib`
  - `Dataset/regression/regression.xlsx`
  - `model/.DS_Store`
  - `model/evaluate.py`
  - `model/train.py`

## Plisson et al. non-hemolytic

- URL: https://github.com/plissonf/ML-guided-discovery-and-design-of-non-hemolytic-peptides
- HEAD SHA: `727fe2d149725ab0922dfc647bbd76c588bc3b72`
- License: MIT
- Target: binary non-hemolytic classification (HemoPI benchmarks)
- Files in tree: 108
- Checkpoint candidate in repo: True
- Local checkpoint present: False
- Inference reproducible: True
- Potentially contaminated vs our test: False
- Status: ok
- Notes: MIT. .pkl classifiers trained on HemoPI sets.
- Interesting files (first 40):
  - `Data/HAMP.fasta`
  - `Data/HemoPI1.fasta`
  - `Data/HemoPI2.fasta`
  - `Data/HemoPI3.fasta`
  - `Data/Hemolytik_datasets/HemoPI-1/HemoPI-1_model_class0.fasta`
  - `Data/Hemolytik_datasets/HemoPI-1/HemoPI-1_model_class1.fasta`
  - `Data/Hemolytik_datasets/HemoPI-1/HemoPI-1_validation_class0.fasta`
  - `Data/Hemolytik_datasets/HemoPI-1/HemoPI-1_validation_class1.fasta`
  - `Data/Hemolytik_datasets/HemoPI-2/HemoPI-2_model_class0.fasta`
  - `Data/Hemolytik_datasets/HemoPI-2/HemoPI-2_model_class1.fasta`
  - `Data/Hemolytik_datasets/HemoPI-2/HemoPI-2_validation_class0.fasta`
  - `Data/Hemolytik_datasets/HemoPI-2/HemoPI-2_validation_class1.fasta`
  - `Data/Hemolytik_datasets/HemoPI-3/HemoPI-3_model_class0.fasta`
  - `Data/Hemolytik_datasets/HemoPI-3/HemoPI-3_model_class1.fasta`
  - `Data/Hemolytik_datasets/HemoPI-3/HemoPI-3_validation_class0.fasta`
  - `Data/Hemolytik_datasets/HemoPI-3/HemoPI-3_validation_class1.fasta`
  - `Data/RPS.fasta`
  - `Data/complete_df_with_activity.csv`
  - `Data/hemolytic_APD.csv`
  - `Data/selected_inliers.fasta`
  - `Data/selected_inliers_sequences.csv`
  - `Data/total_APD.csv`
  - `Data/total_APD.fasta`
  - `Data/total_APD_modified.csv`
  - `Descriptors/HemoPI1_model.csv`
  - `Descriptors/HemoPI1_validation.csv`
  - `Descriptors/HemoPI2_model.csv`
  - `Descriptors/HemoPI2_validation.csv`
  - `Descriptors/HemoPI3_model.csv`
  - `Descriptors/HemoPI3_validation.csv`
  - `Descriptors/generated_inliers.fasta`
  - `Descriptors/generated_inliers_props.csv`
  - `Descriptors/global_descriptors_HemoPI1nm.csv`
  - `Descriptors/global_descriptors_HemoPI1nv.csv`
  - `Descriptors/global_descriptors_HemoPI1pm.csv`
  - `Descriptors/global_descriptors_HemoPI1pv.csv`
  - `Descriptors/global_descriptors_HemoPI2nm.csv`
  - `Descriptors/global_descriptors_HemoPI2nv.csv`
  - `Descriptors/global_descriptors_HemoPI2pm.csv`
  - `Descriptors/global_descriptors_HemoPI2pv.csv`

## HemoNet

- URL: https://github.com/adibayaseen/HemoNet
- HEAD SHA: `ba1c948913b44093f1dae5773bb5ce5833a1d0bc`
- License: NOASSERTION
- Target: binary hemolysis (DBAASP+Hemolytik)
- Files in tree: 17
- Checkpoint candidate in repo: False
- Local checkpoint present: False
- Inference reproducible: False
- Potentially contaminated vs our test: True
- Status: ok
- Notes: weights.hdf referenced; license not declared in README.
- Interesting files (first 40):
  - `DRAMP_Clinical_data.txt`
  - `Datasets.docx`
  - `Ourmodel_UMAP.png`
  - `Supplementary data.docx`
  - `options.json`
  - `weights.hdf`

## PeptideBERT

- URL: https://github.com/ChakradharG/PeptideBERT
- HEAD SHA: `c6a9a8c40f9a0f1edb8b6a1eec7eea3ad46d0311`
- License: MIT
- Target: hemolysis + solubility + non-fouling (task-specific heads)
- Files in tree: 16
- Checkpoint candidate in repo: False
- Local checkpoint present: False
- Inference reproducible: False
- Potentially contaminated vs our test: True
- Status: ok
- Notes: MIT. HuggingFace/PyTorch ecosystem.
- Interesting files (first 40):
  - `data/analysis.py`
  - `data/convert_encodings.py`
  - `data/dataloader.py`
  - `data/dataset.py`
  - `data/download_data.py`
  - `data/split_augment.py`
  - `model/network.py`
  - `model/utils.py`

## peptide-dashboard / MahLooL

- URL: https://github.com/ur-whitelab/peptide-dashboard
- HEAD SHA: `b02d011a1ee64887e663730b66fdc2c1e92a0c43`
- License: GPL-3.0
- Target: E. coli soluble-expression proxy (PROSO-II labels)
- Files in tree: 58
- Checkpoint candidate in repo: True
- Local checkpoint present: False
- Inference reproducible: True
- Potentially contaminated vs our test: False
- Status: ok
- Notes: GPL-3.0. Endpoint mismatch with aqueous synthetic-peptide solubility.
- Interesting files (first 40):
  - `ml/data/hemo-negative.npz`
  - `ml/data/hemo-positive.npz`
  - `ml/data/human-negative.npz`
  - `ml/data/human-positive.npz`
  - `ml/data/insoluble.npz`
  - `ml/data/soluble.npz`
  - `models/hemo-rnn/card.json`
  - `models/hemo-rnn/group1-shard1of1.bin`
  - `models/hemo-rnn/keras_model/model.json`
  - `models/hemo-rnn/keras_model/model_weights.h5`
  - `models/hemo-rnn/model.json`
  - `models/human-rnn/card.json`
  - `models/human-rnn/group1-shard1of1.bin`
  - `models/human-rnn/keras_model/model.json`
  - `models/human-rnn/keras_model/model_weights.h5`
  - `models/human-rnn/model.json`
  - `models/sol-rnn/card.json`
  - `models/sol-rnn/group1-shard1of1.bin`
  - `models/sol-rnn/keras_model/model.json`
  - `models/sol-rnn/keras_model/model_weights.h5`
  - `models/sol-rnn/model.json`
  - `package-lock.json`
  - `package.json`
  - `src/components/lib/milton.json`
  - `src/components/lib/tf-models.js`
  - `src/components/results/ModelCard.vue`

## LysePred

- URL: https://github.com/lincubator/LysePred
- HEAD SHA: `3d202a4228698ac36377ba0208e30b6f0d8e2d33`
- License: NOASSERTION
- Target: binary hemolysis, multiscale CNN
- Files in tree: 98
- Checkpoint candidate in repo: True
- Local checkpoint present: False
- Inference reproducible: True
- Potentially contaminated vs our test: True
- Status: ok
- Notes: Repo oriented to retraining; no released trained checkpoint in root.
- Interesting files (first 40):
  - `data/HemoPI1-1.fasta`
  - `data/HemoPI2-1.fasta`
  - `data/HemoPI3-1.fasta`
  - `data/hlppredfuse-1.fasta`
  - `data/rathore.fasta`
  - `data/rathore2025_external.fasta`
  - `data/rnnamp-1.fasta`
  - `data/rnnamp-origin-1.fasta`
  - `data/statistic/backup_tokendic/proteintoken2index.pkl`
  - `data/statistic/proteintoken2index.pkl`
  - `data/statistic/token2index.py`
  - `frame/DataManager.py`
  - `frame/ModelManager.py`
  - `model/Bert.py`
  - `model/BertGCN/.gitignore`
  - `model/BertGCN/BertGCN_model/__init__.py`
  - `model/BertGCN/BertGCN_model/graphconv_edge_weight.py`
  - `model/BertGCN/BertGCN_model/models.py`
  - `model/BertGCN/BertGCN_model/torch_gat.py`
  - `model/BertGCN/BertGCN_model/torch_gcn.py`
  - `model/BertGCN/README.md`
  - `model/BertGCN/__init__.py`
  - `model/BertGCN/build_graph.py`
  - `model/BertGCN/pretrain`
  - `model/BertGCN/requirements.txt`
  - `model/BertGCN/utils.py`
  - `model/BiLSTM.py`
  - `model/DNAbert.py`
  - `model/DNN.py`
  - `model/ERNIE.py`
  - `model/Focal_Loss.py`
  - `model/GRU.py`
  - `model/LSTM.py`
  - `model/LSTMwithAttention.py`
  - `model/LinformerEncoder.py`
  - `model/LysePred-no-bn.py`
  - `model/LysePred-no-padding.py`
  - `model/LysePred-textcnn.py`
  - `model/LysePred.py`
  - `model/MutiRM.py`

## ML_Peptide (SGF/SIF)

- URL: https://github.com/FrankWanger/ML_Peptide
- HEAD SHA: `9d23d12b41b0d3532bce66934cc6f3bd810a0714`
- License: NOASSERTION
- Target: SGF/SIF stability classification
- Files in tree: 7
- Checkpoint candidate in repo: False
- Local checkpoint present: True
- Inference reproducible: True
- Potentially contaminated vs our test: False
- Status: ok
- Notes: Vendored under external/ml_peptide (SIF_model). No SPDX license.
- Interesting files (first 40):
  - `Sample_Sequence.csv`
  - `model/SGF_model`
  - `model/SIF_model`

## AmpLyze

- URL: None
- HEAD SHA: `None`
- License: n/a
- Target: quantitative HC50 (ProtT5/ESM2 + cross-attention)
- Files in tree: 0
- Checkpoint candidate in repo: False
- Local checkpoint present: False
- Inference reproducible: False
- Potentially contaminated vs our test: False
- Status: methodological_reference_only
- Notes: No reliably identified official code/checkpoint repo. Methodological reference only.
