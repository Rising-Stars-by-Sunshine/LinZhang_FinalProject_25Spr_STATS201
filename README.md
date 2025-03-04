# LinZhang_FinalProject_25Spr_STATS201

## Wavelet-Enhanced Deblurring Diffusion Model (WEDDM) on GS-Blur Dataset

## Disclaimer
This project, _Wavelet-Enhanced Deblurring Diffusion Model (WEDDM) on GS-Blur Dataset_, is the Final Project for **STATS 201: Introduction to Machine Learning for Social Science**, instructed by **Professor Luyao Zhang** at Duke Kunshan University.

## Repository Structure
- **data/**: Contains the dataset, including blurry and sharp images.
- **code/**: Contains the Python code, including Jupyter Notebooks for models training (`Week2ReflectionCode.ipynb`), results visualization (`ProblemSet2.ipynb`), and appendix (`Appendix.ipynb).
- **poster**: For outcome introduction purpose.
- **README.md**: This file, containing setup instructions.

## Prerequisites

### General Requirements
- **Python 3.8 or higher**
- **Google Colab** (Cloud environment setup)
- **Internet connection** for installing dependencies

### Python Dependencies--Required Libraries
Before running the provided Jupyter Notebook Python scripts, ensure that the following libraries are installed on your computer. The following Python libraries are necessary for executing the provided scripts:

#### Core Libraries
`os` – For file and directory handling\
`sys` – For modifying the system path (used to import external scripts)\
`random` – For generating random numbers and randomizing data\
`numpy` – For numerical computations and matrix operations\
`pandas` – For handling and analyzing datasets\
`matplotlib.pyplot` – For visualizing results through plots\
`seaborn` – For enhanced data visualization

#### Deep Learning & Machine Learning Libraries
`torch` – PyTorch framework for deep learning\
`torch.nn` – Neural network module in PyTorch\
`torch.optim` – Optimization algorithms for training deep learning models\
`torch.utils.data` – Data handling utilities in PyTorch\
`torchvision.datasets` – Preloaded datasets for deep learning\
`torchvision.transforms` – Image transformations for preprocessing\
`lpips` – Learned Perceptual Image Patch Similarity (LPIPS) metric for evaluating image restoration quality\
`sklearn.model_selection.train_test_split` – Splitting datasets into training and validation sets\
`sklearn.metrics.mean_squared_error` – For computing Mean Squared Error (MSE)

#### Image Processing Libraries
`PIL` (Pillow) – For opening, manipulating, and saving image files\
`ImageFilter` (from PIL) – For applying image filtering effects, such as motion blur

#### Evaluation Metrics
`skimage.metrics.peak_signal_noise_ratio` (psnr) – Computes PSNR, a metric for image quality\
`skimage.metrics.structural_similarity` (ssim) – Computes SSIM, a perceptual similarity metric

#### Causal Inference and Statistical Analysis
`rdrobust` – For conducting Regression Discontinuity (RD) analysis\
`rdplot` (from `rdrobust`) – For visualizing RD results

#### External Model Dependencies
`MPRNet` – Ensure that the MPRNet repository is cloned and set up correctly. \
\
To install all necessary dependencies, run the following command in your terminal:
```python
pip install torch torchvision numpy pandas matplotlib seaborn pillow scikit-image lpips rdrobust
```
For MPRNet, clone the repository and ensure it is properly configured:
```python
git clone https://github.com/swz30/MPRNet.git
cd MPRNet
```
\
After installing these libraries, you should be able to run the Jupyter Notebook scripts without missing dependencies.




