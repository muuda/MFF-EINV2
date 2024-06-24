# MFF-EINV2
This repo hosts the code and models of "[MFF-EINV2: Multi-scale Feature Fusion across Spectral-Spatial-Temporal Domains for Sound Event Localization and Detection](https://arxiv.org/abs/2406.08771)" [Accepted by Interspeech 2024].



![mff_einv2](/images/mff_einv2.png)

## Data Preparation

- The STARSS22 and STARSS23 datasets can be downloaded from the link.
    - [STARSS22](https://zenodo.org/records/6600531)
    - [STARSS23](https://zenodo.org/record/7709052)
- The official Synthetic dataset can be downloaded from the link. 
    - [Synthetic dataset ](https://zenodo.org/record/6406873#.ZEjVc3ZByUl)

Download and unzip the datasets, the directory of datasets looks like:

```
./dataset
│
├── STARSS22
│   ├── foa_dev
│   ├── foa_eval
│   └── metadata_dev
│
├── STARSS23
│   ├── foa_dev
│   ├── foa_eval
│   └── metadata_dev
│
└── synth_dataset
    └── official
        ├── foa
        └── metadata
```

## Environments

Use the provided `environment.yaml`. Please change the last line of `environment.yaml` to your own Anaconda envs folder and run

```bash
conda env create -f environment.yaml
```

Then activate the environment

```bash
conda activate mff-einv2
```

## Quick Start

Hyper-parameters are stored in `./configs/ein_seld/seld.yaml`. You need to set `dataset_dir` to your own dataset directory. 

The default setting is to only use the STARSS22 and official Synthetic datasets. If you want to use STARSS23 and official Synthetic datasets, please uncomment the commented code in the file `./scripts/preprocess.sh`, and modify `dataset`  parameter to `dcase2023task3` in the file `./configs/ein_seld/seld.yaml`.

### 1. Preprocessing

After downloading the dataset, directly run

```bash
bash ./scripts/preprocess.sh
```

### 2. Training

You can modify the hyper-parameters in `./configs/ein_seld/seld.yaml`. and run

```bash
bash ./scripts/train.sh
```

You can find the training results in the directory `./results/out_train`.

### 3. Inference

The prediction results and model outputs will be saved in the `. /result/out_infer` folder.

```bash
bash ./scripts/infer.sh
```

### 4. Evaluation

Evaluate the generated submission result. Directly run

```bash
python3 code/compute_seld_metrics.py --dataset='STARSS22'
```

or

```bash
python3 code/compute_seld_metrics.py --dataset='STARSS23'
```

## Citation

```
@article{mu2024mffeinv2,
      title={MFF-EINV2: Multi-scale Feature Fusion across Spectral-Spatial-Temporal Domains for Sound Event Localization and Detection}, 
      author={Da Mu and Zhicheng Zhang and Haobo Yue},
      journal={arXiv preprint arXiv:2406.08771},
      year={2024},
}
```

## Reference

The code is based on the [Jinbo-Hu's repo](https://github.com/Jinbo-Hu/DCASE2022-TASK3).

```
@inproceedings{hu2022,
  author={Hu, Jinbo and Cao, Yin and Wu, Ming and Kong, Qiuqiang and Yang, Feiran and Plumbley, Mark D. and Yang, Jun},
  booktitle={ICASSP 2022 - 2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={A Track-Wise Ensemble Event Independent Network for Polyphonic Sound Event Localization and Detection}, 
  year={2022},
  pages={9196-9200},
  doi={10.1109/ICASSP43922.2022.9747283}}
```

