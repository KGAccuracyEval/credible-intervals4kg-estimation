# Credible Intervals for KG Accuracy Estimation
Knowledge Graphs (KGs) are widely used in data-driven applications and downstream tasks, such as virtual assistants, recommendation systems, and semantic search. The accuracy of KGs directly impacts the reliability of the inferred knowledge and outcomes. Therefore, assessing the accuracy of a KG is essential for ensuring the quality of facts used in these tasks. However, the large size of real-world KGs makes manual triple-by-triple annotation impractical, thereby requiring sampling strategies to provide accuracy estimates with statistical guarantees.
The current state-of-the-art approaches rely on Confidence Intervals (CIs), derived from frequentist statistics. While efficient, CIs have notable limitations and can lead to interpretation fallacies. In this paper, we propose to overcome the limitations of CIs by using *Credible Intervals* (CrIs), which are grounded in Bayesian statistics. These intervals are more suitable for reliable post-data inference, particularly in KG accuracy evaluation. We prove that CrIs offer greater reliability and stronger guarantees than frequentist approaches in this context. Additionally, we introduce *a*HPD, an adaptive algorithm that is more efficient for real-world KGs and statistically robust, addressing the interpretive challenges of CIs.

## Contents
This repository contains code, data, proofs, and extra experiments for the paper "Credible Intervals for Knowledge Graph Accuracy Estimation" accepted at SIGMOD 2025.

## Installation

1. Install Python 3.10 if not already installed. <br>
2. Install the required dependencies by running:

```bash
pip install -r requirements.txt
```

## Datasets

The datasets used in the experiments are YAGO, NELL, DBPEDIA, FACTBENCH, and SYN (100M). <br>
Each dataset can be accessed in its corresponding directory: ```/dataset/{YAGO|NELL|DBPEDIA|FACTBENCH|SYN}/```. 

In order to use the SYN 100M dataset, it must be generated first. <br>
To do so, move to ```/dataset/SYN/generate/``` and then execute:

```bash
python generateGraph.py
```

## Running Experiments

For each dataset, experiments can be executed using the corresponding script. The default confidence level is set to ```0.05```, but this can be adjusted by changing the ```--confLevel``` parameter. The confidence levels considered in the paper are ```{0.10, 0.05, 0.01}```.

To run experiments on YAGO, run:

```bash
bash runYAGO.sh
```

To run experiments on NELL, run:

```bash
bash runNELL.sh
```

To run experiments on DBPEDIA, run:

```bash
bash runDBPEDIA.sh
```

To run experiments on FACTBENCH, run:

```bash
bash runFACTBENCH.sh
```

To run experiments on SYN (100M), run:

```bash
bash runSYN.sh
```

To experiment with different configurations, directly use ```runKGEval.py```. The description of all the available arguments can be obtained by running:

```bash
python runKGEval.py --help
```

## Extra Experiments

Extra experimental results presenting additional sampling strategies are available in the ```/extra-experiments/``` folder. These results are consistent with those presented in the paper, but are provided here due to space limitations.

## Complete Proofs

The complete proofs for Theorems 1-3 can be found in the ```/complete-proofs/``` folder. These full proofs are provided here and not in the paper for space reasons.

## Acknowledgments

The work was supported by the HEREDITARY project, as part of the EU Horizon Europe program under Grant Agreement 101137074.
