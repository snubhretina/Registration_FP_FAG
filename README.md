# Registration_FP_FAG
# Fully Leveraging Deep Learning Methods for Constructing Retinal Fundus Photomontages


## Introduction
This repository described in the paper "Fully Leveraging Deep Learning Methods for Constructing Retinal Fundus Photomontages" (https://ieeexplore.ieee.org/abstract/document/9050794)
![image](https://user-images.githubusercontent.com/64057617/156965423-d221dfdf-4f23-4339-8863-950da048c9fa.png)
## Usage

### Installation
```
git clone snubhretina/Registration_FP_FAG
cd Registration_FP_FAG
```

* Download the pretrained Vessel extraction models form [here]. This model is trained DRIVE Database. Our model can't provide cause trained our SNUBH internel DB.
* Unzip and move the pretrained parameters to models/

### Run
```
python main.py --input_path="./data" --output_path="./res/" --fp_model_path = "./model/seg_model.pth" --fag_model_path = "./model/seg_model.pth"
```
you must input model


## Citation
```
@article{noh2020multimodal,
  title={Multimodal registration of fundus images With fluorescein angiography for fine-scale vessel segmentation},
  author={Noh, Kyoung Jin and Kim, Jooyoung and Park, Sang Jun and Lee, Soochahn},
  journal={IEEE Access},
  volume={8},
  pages={63757--63769},
  year={2020},
  publisher={IEEE}
}
```
