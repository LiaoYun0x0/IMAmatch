# IMAMatch

### Semi-dense Feature matching with Increased Matching Amount

### Introduction

In this work, we mainly propose a method of adding matching points in the refinement matching stage. It is very effective in improving matching accuracy.

### Installation

```bash
conda create -n IMAMatch python=3.10
conda activate IMAMatch
cd /path/to/IMAMatch
pip install -r requirement.txt
```

### Datasets

We use the same train/test dataset as [LoFTR](https://github.com/zju3dv/LoFTR/blob/master/docs/TRAINING.md).
place the dataset and index in the data directory.
A structure of dataset should be:

```
megadepth_root_dir
├── phoenix
├── Undistorted_SfM
└── megadepth_indices

scannet_root_dir
├── scannet_all
│   ├── scene0000_00
│   ├── ...
│   └── scene0806_00
└── scanent_indices   
     ├── intrinsics.npz
     └── scene_data
```
Then set your megadepth_root_dir and scannet_root_dir in configs/data_config.py



### Train and Evaluation

For both train and evaluation, you can just excute the corresponding scripts. To set the parameters, just modity the cfg dict inner the script. 



```
@inproceedings{IMAMatch,
  title={Semi-dense Feature matching with Increased Matching Amount},
  author={Wang, Qing and Zhang, Jiaming and Yang, Kailun and Peng, Kunyu and Stiefelhagen, Rainer},
  booktitle={Asian Conference on Computer Vision},
  year={2022}
}
```

### Acknowledgments

Our work is based on [LoFTR](https://github.com/zju3dv/LoFTR) and we use their code.  We appreciate the previous open-source repository [LoFTR](https://github.com/zju3dv/LoFTR).
