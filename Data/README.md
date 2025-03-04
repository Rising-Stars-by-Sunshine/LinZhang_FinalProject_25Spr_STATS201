# Data

This folder contains the dataset used for training and testing deblurring algorithms. It includes both blurry input images and their corresponding clear target images. The dataset is designed to assist in research on deblurring techniques by providing paired examples of noisy (blurry) and clean images.

## Dataset Overview

**Note:** These datasets are too large to upload to the GitHub folder. You can access the data using the following link:

**This is the official implementation of GS-Blur: A 3D Scene-Based Dataset for Realistic Image Deblurring (NeurIPS 2024).**  
[Download the dataset here](https://drive.google.com/drive/folders/1ZksD7bPl3_ezDLoeHJ2Duwo_LQG1TXB1).
Please download mini.zip

This dataset is used in the paper:

Lee, Dongwoo, Joonkyu Park, and Kyoung Mu Lee. 2024. “GS-Blur: A 3D Scene-Based Dataset for Realistic Image Deblurring.” *ArXiv.org*. 2024. [https://arxiv.org/abs/2410.23658](https://arxiv.org/abs/2410.23658).

The dataset is designed to assist in the development and evaluation of deblurring algorithms, offering a diverse set of synthetic blur types created from 3D scene-based simulations. The data includes paired blurry and clear images for training and testing deblurring models, with various types of blur such as motion blur, defocus blur, and camera shake.


The dataset consists of two main subfolders:

1. **input_noise**: Contains 1001 blurry images.
2. **target**: Contains 1001 corresponding sharp (clear) images for each blurry image in the `input_noise` folder.

This dataset can be used for training and evaluating machine learning models for image deblurring tasks.

## Data Source

The dataset is synthetic and generated using a novel approach to simulate various types of blur, including motion blur, defocus blur, and camera shake. These blurry images serve as realistic examples of common image degradation found in real-world scenarios.

## Data Dictionary

| Variable Name      | Description                                                                 |
|--------------------|-----------------------------------------------------------------------------|
| `input_noise/*`     | Blurry images that serve as the input for deblurring models.                |
| `target/*`          | Sharp images corresponding to each blurry image in the `input_noise` folder. |

## Instructions for Use

1. **Data Structure**: The data is organized into two folders, `input_noise` and `target`. The images in these folders are paired by filename, where the blurry image in `input_noise` corresponds to the clear image in `target`.

2. **Data Size**: The dataset contains a total of 1001 image pairs.

