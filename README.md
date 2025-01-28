# Scoliosis_project
This is an official repo for the paper on automatic scoliosis angle measurement: [Deep learning automates Cobb angle measurement compared with multi-expert observers](https://arxiv.org/abs/2403.12115).
<img width="937" alt="graphical_abstract" src="https://github.com/user-attachments/assets/afd18869-7306-4ec9-a7e0-498438682f51" />

Provided by: Hanxue Gu

# Installation
## 1. Create and Activate a Conda Environment

```bash
conda create -n spine-env python=3.8 -y
conda activate spine-env

conda install pytorch==1.7.0 torchvision==0.8.1 cudatoolkit=11.0 -c pytorch

# With custom CUDA ops
pip install mmdet==2.16.0
pip install mmcv-full==1.3.16

# OR, basic version (no custom CUDA ops):
# pip install mmdet==2.16.0
# pip install mmcv==1.3.16

pip install pycocotools matplotlib scikit-image jupyterlab
```

## 2. Verify your installation
```bash
python -c "
import torch, torchvision, mmcv, mmdet
print('Torch:', torch.__version__)
print('TorchVision:', torchvision.__version__)
print('MMCV:', mmcv.__version__)
print('MMDetection:', mmdet.__version__)
```

# Data Preparation

1. **Inference:** Place your sample image (in PNG format) under `./example_case`.
2. **Training/Validation:** Organize your dataset into `./train` and `./val` folders, and provide additional JSON files that reference these images. Examples of the JSON format can be found in:
3. 
Examples of json format can be found in:
```
raw_train_label_new.json
raw_val_label_new.json
```

# Train/Validate the Model

Use the notebook [`train_and_val.ipynb`](train_and_val.ipynb) for model training and validation.

# Inference/Test on a Single Case with Visualization

Use the notebook [`inference_1_case.ipynb`](inference_1_case.ipynb) to perform inference on a single case. The results, including the main Cobb angle annotation, will be displayed step by step.


# Comments
We build out a detection algorithm mainly based on MMDetection. The details of MMDetection can be found [here](https://github.com/open-mmlab/mmdetection).
If you find our repo helpful, please feel free to cite our work:
```
@misc{li2024deeplearningautomatescobb,
      title={Deep learning automates Cobb angle measurement compared with multi-expert observers}, 
      author={Keyu Li and Hanxue Gu and Roy Colglazier and Robert Lark and Elizabeth Hubbard and Robert French and Denise Smith and Jikai Zhang and Erin McCrum and Anthony Catanzano and Joseph Cao and Leah Waldman and Maciej A. Mazurowski and Benjamin Alman},
      year={2024},
      eprint={2403.12115},
      archivePrefix={arXiv},
      primaryClass={eess.IV},
      url={https://arxiv.org/abs/2403.12115}, 
}
```
