# comp-geno-proj
Repo for CBMFW4761 2023 Spring final project: predicting gene expression based on epigenetics and chromatin structures

Contributers: 
C. Angel, J. Blindenbach, F.Shiau

## Directory structure:
```
├── README.md
└── src
    ├── 1_preprocess
    │   └── 1.0_pre_processing.sh
    ├── 2_hmm
    │   ├── 2.1_split-data-hmm.py
    │   ├── 2.2_gmm-select.ipynb
    │   ├── 2.3_gmm-fit.ipynb
    │   └── 2.4_gmm-hmm.ipynb
    └── 3_ml
        ├── README.md
        ├── binary.ipynb
        ├── merged_allData_promoters.csv
        ├── promoter.ipynb
        └── signal.ipynb
```
## 1_preprocess
Preprocesses the data to be used for training and testing

### Requirements
The raw data files are not included in this repo. They can be downloaded from ENCODE portal (https://www.encodeproject.org/). 

`bedtools` and `kentUtils` to be installed.

File was written for NYGC Slurm system. Modify the paths to the data files and output files to run on other systems.
### Output Files
Files Description: 
5kb windows approach
https://drive.google.com/drive/folders/1tfv7ZjXAd2Up2wRdaYA2Pfqan7Tt2zUQ?usp=share_link
Subdirectories:
    - RNA-pol
    - HiC_AB
    - H3K27ac
    - H3K4me1
    - H3K4me2
    - H3K9Ac
    - H3K4me3
    - H3K27me3
    - ATAC-seq 

Each folder contains the processed 5kb genome wide windows bedGraph files. Their name refers to the replicate accession number.

Promoters approach
https://github.com/chrlosangel/ComputationalGenomics_project/promoters_data
Subdirectories:
    - RNA-pol
    - HiC_AB
    - H3K27ac
    - H3K4me1
    - H3K4me2
    - H3K9Ac
    - H3K4me3
    - H3K27me3
    - ATAC-seq 

Each folder contains the processed bedGraph files with signal in all promoters annotated in the hg19 human genome version. Their name refers to the replicate accession number.


## 2_hmm
Implements the GMM-HMM model and its training

### Requirements
`python3` and `jupyter notebook` to be installed.

Packages used:
```
3.10.10 | packaged by conda-forge | (main, Mar 24 2023, 20:12:31) [Clang 14.0.6 ]
numpy: 1.23.5
torch: 2.0.0
sklearn: 1.2.2
pandas: 2.0.1
scipy: 1.10.1
lightning.pytorch: 2.0.1.post0
os
sys
glob
matplotlib
seaborn
collections
tqdm
```
### To run:
1. Run `2.1_split-data-hmm.py` to split the data into training and testing sets. Modify the path to correct data files in the script and an existing output path.
2. Run `2.2_gmm-select.ipynb` to select the number of components for the GMM model. Modify the path to use the correct output files from `2.1_split-data-hmm.py`.
3. Run `2.3_gmm-fit.ipynb` to fit the GMM model and visually inspect how each state clusters the genome. Modify the path to use the correct output files from `2.1_split-data-hmm.py`.
4. Run `2.4_gmm-hmm.ipynb` to train the GMM-HMM model and predict RNA Pol II binding. Modify the path to use the correct training/testing data from `2.1_split-data-hmm.py`.

## 3_ml:  ML Predictions
This directory contains our ML solution contained in three different notebooks corresponding to the three different types of data we used, `binary.ipynb` for binary data, `signal.ipynb` for signal data, and `promoter.ipynb` for signal data only in the promoter region.

Requirements to run the jupyter notebooks,

- PyTorch
- SkLearn
- MatPlotLib
- Pandas