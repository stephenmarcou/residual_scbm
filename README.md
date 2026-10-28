# Residual Stochastic Concept Bottleneck Models

Code for the paper *Residual Stochastic Concept Bottleneck Models* (Res-SCBM).

Res-SCBMs model concept logits and residual logits jointly through a single multivariate
Gaussian, so a test-time concept intervention propagates to statistically dependent
residual variables through Gaussian conditioning. An intervention-aware loss `L_int`
keeps the model responsive to concept corrections as residual capacity grows.

This repository builds on the public implementation of *Stochastic Concept Bottleneck
Models*; see `LICENSE.md` and *Acknowledgements* below.

---

## 1. Setup

```bash
conda env create -f environment.yml
conda activate scbm
```

Three paths can be pointed outside the repository with environment variables; each falls
back to the value in the configs, so a self-contained checkout needs none of them:

| Variable | Default | What it holds |
|---|---|---|
| `SCBM_DATA_PATH` | `datasets/` (from `configs/data/data_defaults.yaml`) | the datasets |
| `SCBM_EXPERIMENT_DIR` | `./experiments/` | run folders |
| `SCBM_MODEL_DIRECTORY` | `./pretrained_networks/` | pretrained backbones |

The CUB models use an ImageNet-pretrained ResNet-18. If no checkpoint is found under
`SCBM_MODEL_DIRECTORY`, torchvision's own `IMAGENET1K_V1` weights are downloaded
instead, so no manual step is required.

## 2. Data

**CUB.** Raw CUB images are not redistributed in this repository. Download
[CUB-200-2011](https://www.vision.caltech.edu/datasets/cub_200_2011/) from the
official dataset page, extract it, and place the image directory at:

```text
datasets/CUB/CUB_200_2011/CUB_200_2011/images/
```

The label side is included: the 112-concept class-level split from the original
*Concept Bottleneck Models* paper at
`datasets/CUB/class_attr_data_10/`, plus `attributes.txt` and `concept_names.txt`.

**CUB-Incomplete** is included, at `datasets/CUB/incomplete_data/incomplete_cub/`: the 22
concepts across 7 groups that remain after retaining 25% of the concept groups, as in
*Learning to Receive Help: Intervention-Aware Concept Embedding Models*. Its `info.txt` records the removed group names, the
removed attribute indices and the old→new index mapping. It is shipped rather than
regenerated because the retained groups are a fixed set, not a random draw.

**AwA2.** Raw AwA2 images are not redistributed in this repository. Download the
[AwA2 dataset and attributes](https://cvml.ist.ac.at/AwA2/) from the official dataset
page and extract `JPEGImages/` and `classes.txt` under `datasets/AwA2/`.

The AwA2 models use precomputed 2048-dimensional
ResNet-101 embeddings rather than pixels. Download the `xlsa17` feature release from the official dataset [AwA2 dataset and attributes](https://cvml.ist.ac.at/AwA2/) 
and place `res101.mat` at:

```text
datasets/AwA2/xlsa17/data/AWA2/res101.mat
```

Alternatively, set `data.res101_path` to the location of that file.

The 60/20/20 image-level split manifest (`split_info.json`), `classes.txt`,
`predicates.txt` and the 9-predicate **AwA2-Incomplete** concept set
(`datasets/AwA2/incomplete_data/awa2_incomplete_local_custom_1/`) are included here.

## 3. Training

Every argument is a Hydra override. `+model=` and `+data=` select the config groups.

```bash
# Res-SCBM (with L_int) on CUB-Incomplete, R = 20
python train.py +model=SCBM_RES +data=CUB seed=0 \
    incomplete=True data.pkl_file_dir=incomplete_cub/ \
    data.num_residuals=20 model.use_L_int_extension_loss=True \
    model.inter_strategy=emp_perc save_model=True

# Res-SCBM w/o L_int: drop the loss flag
# SCBM / Hard CBM / CEM / black-box DNN:
python train.py +model=SCBM  +data=CUB seed=0 incomplete=True data.pkl_file_dir=incomplete_cub/
python train.py +model=CBM   +data=CUB seed=0 incomplete=True data.pkl_file_dir=incomplete_cub/ model.inter_strategy=hard
python train.py +model=CEM   +data=CUB seed=0 incomplete=True data.pkl_file_dir=incomplete_cub/ model.inter_strategy=hard model.p_int=0.25
python train.py +model=DNN   +data=CUB seed=0 incomplete=True data.pkl_file_dir=incomplete_cub/

# Linear head on ground-truth concepts (the reference row of Table 1)
python eval_datasets.py +model=target_head +data=CUB seed=0 incomplete=True data.pkl_file_dir=incomplete_cub/
```

Swap `+data=AwA2` and `data.pkl_file_dir=awa2_incomplete_local_custom_1` for the AwA2
tasks, and drop `incomplete=True` for the complete ones.

**Intervention strategy (`model.inter_strategy`):**
- Res-SCBM with L_int: use `emp_perc`. The L_int loss intervenes during training
  with this strategy.
- Hard CBM and CEM: use `hard`, which sets each intervened concept to 0 or 1.

The paper reports three seeds,
`seed=0`, `10` and `20`. Run the Python entry points directly from the repository root;
for example, use `python train.py` for training and `python inference.py` for
post-hoc evaluation.

A run writes to `experiments/<model>/<dataset>/<auto-named-run>/`. The **first line of
`log.txt` is the full config** of the run, which is how `inference.py` recovers a
checkpoint's setup.

## 4. Interventions

Intervention curves are produced post-hoc from a saved checkpoint:

```bash
python inference.py +model=SCBM_RES +data=CUB \
    inference.ex_name=<run_folder_name> incomplete=True \
    run_inference=True run_interventions=True \
    model.inter_strategy=<strategy>
```

Set `<strategy>` as in the training section: `emp_perc` for Res-SCBM with L_int, `hard`
for CBM and CEM.

This writes `inference_log.txt` (test-set metrics) and `intervention_log.txt` (one line
per intervention step) inside the run folder. With `incomplete=True` the run's concept
set, `num_concepts` and `num_residuals` are restored from its own `log.txt`, so they do
not have to be passed again.

## Acknowledgements

This code extends the official implementation of *Stochastic Concept Bottleneck Models*:

```
@inproceedings{DBLP:conf/nips/VandenhirtzLMV24,
 author = {Vandenhirtz, Moritz and Laguna, Sonia and Marcinkevi\v{c}s, Ri\v{c}ards and Vogt, Julia E.},
 booktitle = {Advances in Neural Information Processing Systems},
 doi = {10.52202/079017-1641},
 editor = {A. Globerson and L. Mackey and D. Belgrave and A. Fan and U. Paquet and J. Tomczak and C. Zhang},
 pages = {51787--51810},
 publisher = {Curran Associates, Inc.},
 title = {Stochastic Concept Bottleneck Models},
 url = {https://proceedings.neurips.cc/paper_files/paper/2024/file/5c7894ac8788555f1cecf536f1e0fd35-Paper-Conference.pdf},
 volume = {37},
 year = {2024}
}
```
