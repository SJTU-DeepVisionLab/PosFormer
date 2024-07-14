# PosFormer

<h3 align="center"> <a href="https://arxiv.org/abs/2407.07764">PosFormer: Recognizing Complex Handwritten Mathematical Expression with Position Forest Transformer</a></h3>


<h5 align="center">

[![arXiv](https://img.shields.io/badge/Arxiv-2407.07764-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2407.07764)

# Description
 This repository provides the official implementation of the Position Forest Transformer (PosFormer) for Handwritten Mathematical Expression Recognition (HMER). This innovative model introduces a dual-task approach, optimizing both expression and position recognition to facilitate position-aware feature learning for symbols in mathematical expressions. It employs a novel structure called a position forest to parse and model the hierarchical relationships and spatial positioning of symbols without the need for extra annotations. Additionally, an implicit attention correction module is integrated into the sequence-based decoder architecture to enhance focus and accuracy in symbol recognition. PosFormer demonstrates significant improvements over existing methods on several benchmarks, including single-line CROHME datasets and more complex multi-line and nested expression datasets, achieving higher performance without extra computational overhead. This repository includes code, pre-trained models, and usage instructions to aid researchers and developers in applying and further developing this state-of-the-art HMER solution.




## News 
* ```2024.7.10 ``` 🚀 [MNE](https://drive.google.com/file/d/1iiCxwt05v9a7jQIf074F1ltYLNxYe63b/view?usp=drive_link) available.


## MNE
The MNE dataset can now be downloaded [here](https://drive.google.com/file/d/1iiCxwt05v9a7jQIf074F1ltYLNxYe63b/view?usp=drive_link).

The Multi-level Nested Expression (MNE) dataset is specifically designed for evaluating the capability of models to recognize complex handwritten mathematical expressions. It comprises three subsets categorized based on the nested levels of the expressions: N1, N2, and N3, representing one, two, and three levels of nesting, respectively. This dataset avoids using the N4 level due to its minimal representation (only 0.2%) in public datasets. The distribution of the other levels in existing datasets includes 37.4% for N1, 51.4% for N2, 9.7% for N3, and 1.3% for other complexities.

Originally, the images for subsets N1 and N2 were sourced from the CROHME test sets and include 1875 and 304 images, respectively. The N3 subset, containing initially only 10 images, was expanded to 1464 images to provide a robust challenge in identifying highly complex expressions. This expansion was achieved by incorporating complex expression images from public documents and real-world handwritten homework, cited from multiple sources in the research literature. 


## Getting Started

### Installation
```bash
cd PosFormer
# install project   
conda create -y -n PosFormer python=3.7
conda activate PosFormer
conda install pytorch=1.8.1 torchvision=0.2.2 cudatoolkit=11.1 pillow=8.4.0 -c pytorch -c nvidia
# training dependency
conda install pytorch-lightning=1.4.9 torchmetrics=0.6.0 -c conda-forge
# evaluating dependency
conda install pandoc=1.19.2.1 -c conda-forge
pip install -e .
```
### Data Preparation

Please download CROHME and MNE datasets and organize them as follows

```
- PosFormer
  | - data.zip
  | - data_MNE.zip

```

### Training

For training, we utilize a single A800 GPU; however, an RTX 3090 GPU also provides sufficient memory to support a training batch size of 8. The training process is expected to take approximately 25 hours on a single A800 GPU.

```bash
cd PosFormer
python train.py --config config.yaml
```

### Evaluation 


```bash
cd PosFormer
perl --version  # make sure you have installed perl 5
# results will be printed in the screen and saved to lightning_logs/version_0 folder
bash eval_all.sh 0
```

 ### TODO
 1. Release the code of PosFormer
 2. Release checkpoint
 3. Improve README and samples
