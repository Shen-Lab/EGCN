# EGCN
Energy-based Graph Convolutional Networks for Scoring Protein Docking Models

![EGCN Architecture](/EGCN_fig1.png)


# Results:
|  Dataset  | Model | Top 1  | Top 5 | Top 10 | 
| :--- |:--- | :--- |:--- | :--- |
| Benchmark test set (107)  | IRAD  |  10/0**/0*** |  30/13**/2*** | 40/18/7*** |
|    | RF  |  13/5**/0*** | 35/17**/5*** | 42/25**/10*** |
|    | EGCN | 17/8**/2*** | 46/21**/5*** | 51/28**/11*** |
|     | Best possible | 70/47**/20*** | 70/47**/20*** | 70/47**/20*** |
| CAPRI test set (14)  | IRAD  |   3/1**/0*** | 4/2**/0*** | 6/3**/1*** |
|    | RF  |   4/1**/1*** | 6/2**/1*** | 8/3/2*** |
|    | EGCN | 5/1**/1*** | 6/3**/2*** | 7/4/2*** |
|     | Best possible |  9/6**/3*** | 9/6**/3*** | 9/6**/3*** |

 For both the benchmark and CAPRI test sets, EGCN significantly outperformed IRAD. First, for the benchmark test set of
107 cases, EGCN generated 46 targets with at least acceptable top-5
predictions (the most possible number being 70 for the decoy set),
representing more than 50% increase compared to 30 achieved by
IRAD. The performance improvement for ranking medium and highquality predictions was even more impressive: EGCN generated
21 and 5 targets with medium and high-quality top-5 predictions,
respectively, representing 62% and 150% increases compared to
IRAD. Second, for the CAPRI test set, although the total number of
targets (14) can be too few to provide statistical significance, EGCN again increased the number of at least acceptable, medium, and highquality top-5 models from IRAD's 4, 2, 0 to 6, 3, 2, respectively.

# Dependencies:
* Tensorflow >= 1.13

# File explanation:
* src/model0_residue_based.py: main source code for EGCN
* src/train.py:  training code 
* feature_gen/feature_gen.py: generate features
* feature_gen/rosetta_param.py: storing Rosetta parameters used in our features
* feature_gen/sasa_cal.py: calculate SASA features

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

## Contact:
Yang Shen: yshen@tamu.edu

Yue Cao:  cyppsp@tamu.edu
