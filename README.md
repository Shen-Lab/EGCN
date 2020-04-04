# EGCN
Energy-based Graph Convolutional Networks for Scoring Protein Docking Models

![EGCN Architecture](/EGCN_fig1.png)

# Dependencies:
* Tensorflow >= 1.13

# File explanation:
* src/model0_residue_based.py: main source code for EGCN
* src/train.py:  training code 
* feature_gen.py: generating features
* rosetta_param.py: storing Rosetta parameters used in our features
* sasa_cal.py: Calculate SASA features

# Training data dependencies:
* [Piper](https://cluspro.bu.edu/downloads.php) 
* [cNMA](https://github.com/Shen-Lab/cNMA)
* [Freesasa](https://freesasa.github.io/)


# Citation:

```
@misc{cao2019energybased,
    title={Energy-based Graph Convolutional Networks for Scoring Protein Docking Models},
    author={Yue Cao and Yang Shen},
    year={2019},
    eprint={1912.12476},
    archivePrefix={arXiv},
    primaryClass={q-bio.BM}
}
```
