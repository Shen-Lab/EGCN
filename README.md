# EGCN
Energy-based Graph Convolutional Networks for Scoring Protein Docking Models

https://onlinelibrary.wiley.com/doi/abs/10.1002/prot.25888

https://www.biorxiv.org/content/10.1101/2019.12.19.883371v1  (earlier version without the table below) 

![EGCN Architecture](/EGCN_fig1.png)



## Abstract 

Structural information about protein‐protein interactions, often missing at the interactome scale, is important for mechanistic understanding of cells and rational discovery of therapeutics. Protein docking provides a computational alternative for such information. However, ranking near‐native docked models high among a large number of candidates, often known as the scoring problem, remains a critical challenge. Moreover, estimating model quality, also known as the quality assessment problem, is rarely addressed in protein docking. 

In this study, the two challenging problems in protein docking are regarded as relative and absolute scoring, respectively, and addressed in one physics‐inspired deep learning framework. We represent protein and complex structures as intra‐ and inter‐molecular residue contact graphs with atom‐resolution node and edge features. And we propose a novel graph convolutional kernel that aggregates interacting nodes’ features through edges so that generalized interaction energies can be learned directly from 3D data. The resulting energy‐based graph convolutional networks (EGCN) with multihead attention are trained to predict intra‐ and inter‐molecular energies, binding affinities, and quality measures (interface RMSD) for encounter complexes. 

Compared to a state‐of‐the‐art scoring function for model ranking, EGCN significantly improves ranking for a critical assessment of predicted interactions (CAPRI) test set involving homology docking; and is comparable or slightly better for Score_set, a CAPRI benchmark set generated by diverse community‐wide docking protocols not known to training data. For Score_set quality assessment, EGCN shows about 27% improvement to our previous efforts. Directly learning from 3D structure data in graph representation, EGCN represents the first successful development of graph convolutional networks for protein docking.


## Results: 

(Relative Scoring / Ranking) 

Following the CAPRI convention, reported are the numbers of targets with at least an acceptable, at most a medium ( ** ), or at most a high-quality ( *** ) model within top 1/5/10 predictions ranked by individual scoring functions.  For instance, 7/4**/2*** means that a scoring function has generated 7 targets with at least acceptable top-ranked models, including 4 with medium and 2 with high-quality predictions at best.

|  Dataset  | Model | Top 1  | Top 5 | Top 10 | 
| :--- |:--- | :--- |:--- | :--- |
| Benchmark test set (107)  | IRAD  |  10/0**/0*** |  30/13**/2*** | 40/18/7*** |
|    | RF  |  13/5**/0*** | 35/17**/5*** | 42/25**/10*** |
|    | EGCN | 17/8**/2*** | 46/21**/5*** | 51/28**/11*** |
|     | Best possible | 70/47**/20*** | 70/47**/20*** | 70/47**/20*** |
| CAPRI test set (14)  | IRAD  |   3/1**/0*** | 4/2**/0*** | 6/3**/1*** |
|    | RF  |   4/1**/1*** | 6/2**/1*** | 8/3/2*** |
|    | EGCN | 5/1**/1*** | 6/3**/2*** | 7/4/2*** |
|    | Best possible |  9/6**/3*** | 9/6**/3*** | 9/6**/3*** |
| CAPRI score_set (13) | IRAD | 3/2**/0*** | 5/4**/1*** | 7/4**/2*** | 
|    | RF | 1/0**/0*** | 3/2**/0*** | 3/2**/1*** |
|    | EGCN | 3/2**/0*** | 6/4**/1*** | 7/4**/1*** | 
|    | Best Possible  | 11/6**/3*** | 11/6**/3*** | 11/6**/3*** | 

(Jan. 19, 2020) Please note the correction in the last row of the published table: the best possible outcomes for the CAPRI score_set were incorrectly reported 11/9**/3*** but should be 11/6**/3*** instead.  

RF refers to our earlier random forest model from https://github.com/Shen-Lab/BAL (dependency 3)


(Absolute Scoring / Quality Estimation)

See Fig. 3 in the Paper 

## Dependencies:
* Tensorflow >= 1.13

## File explanation:
* src/model0_residue_based.py: main source code for EGCN
* src/train.py:  training code 
* feature_gen/feature_gen.py: generate features
* feature_gen/rosetta_param.py: storing Rosetta parameters used in our features
* feature_gen/sasa_cal.py: calculate SASA features

## Training data dependencies:
* [Piper](https://cluspro.bu.edu/downloads.php) 
* [cNMA](https://github.com/Shen-Lab/cNMA)
* [Freesasa](https://freesasa.github.io/)

## Trained models:
* Under trained_models/, we have a trained model.

## Generate Features:
* Prepare a input file. Every two lines store the paths to the receptor and ligand of your pdb complex in your training set (the first line is receptor's path, the second line is ligand's path). An example is like this:
  - ..../data/1ACB_r.pdb
  - ..../data/1ACB_l.pdb
  - ..../data/1DQJ_r.pdb
  - ..../data/1DQJ_l.pdb 
* Go to feature_gen/,  Run 
* `python feat_gen.py  $path_to_your_input_file  $path_to_your_output_directory ` 
* to generate features.

## Training:
* Go to src/, run `python train.py`.

## Evaluation:
* Go to src/, run `python evaluate.py`.

## Citation:

```
@article{EGCN,
author = {Cao, Yue and Shen, Yang},
title = {Energy-based graph convolutional networks for scoring protein docking models},
journal = {Proteins: Structure, Function, and Bioinformatics},
volume = {88},
number = {8},
pages = {1091-1099},
keywords = {energy-based models, graph convolutional networks, machine learning, protein docking, protein-protein interactions, quality estimation, scoring function},
doi = {https://doi.org/10.1002/prot.25888},
url = {https://onlinelibrary.wiley.com/doi/abs/10.1002/prot.25888},
eprint = {https://onlinelibrary.wiley.com/doi/pdf/10.1002/prot.25888},
year = {2020}
}
```

## Contact:
Yang Shen: yshen@tamu.edu

Yue Cao:  cyppsp@tamu.edu
