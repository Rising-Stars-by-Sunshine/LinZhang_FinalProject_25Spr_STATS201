# LinZhang_FinalProject_25Spr_STATS201

## Wavelet-Enhanced Deblurring Diffusion Model (WEDDM) on GS-Blur Dataset

----------------------------------------
## Table of Contents
- [Disclaimer](#disclaimer)
- [Repository Structure](#repository-structure)
- [Prerequisites](#prerequisites)
  - [1. General Requirements](#1-general-requirements)
  - [2. Required Dataset Download](#2-required-dataset-download)
  - [3. Python Dependencies—Required Libraries](#3-python-dependencies--required-libraries)
- [Navigation Instructions](#navigation-instructions)
- [Report](#report)
- [Poster](#poster)
- [Acknowledgements](#acknowledgments)
- [Statement of Intellectual and Professional Growth](#statement-of-intellectual-and-professional-growth)

----------------------------------------

## Disclaimer
This project, _Wavelet-Enhanced Deblurring Diffusion Model (WEDDM) on GS-Blur Dataset_, is the Final Project for **STATS 201: Introduction to Machine Learning for Social Science**, instructed by **Professor Luyao Zhang** at Duke Kunshan University in Autumn 2025.

----------------------------------------

## Repository Structure
- **data/**: Contains the dataset, including blurry and sharp images.
- **code/**: Contains the Python code, including Jupyter Notebooks for models training (`Simulated_WEDDM.ipynb`&`GSBlur_MPRNet.ipynb`&`AddNoise_GSBlur_10WEDDM.ipynb`&`AddNoise_GSBlur_50WEDDM.ipynb`
), results visualization (`ResultsVisualization.ipynb`), and appendix (`Appendix.ipynb`).
- **Report**
- **Poster**: For outcome introduction purpose.
- **README.md**: This file, containing setup instructions.

----------------------------------------

## Prerequisites

### 1. General Requirements
To run the code and reproduce the results, ensure you have the following setup:

- **Python 3.8 or higher**: The code is compatible with Python 3.8 and above.
- **Jupyter Notebook**: All experiments and visualizations are implemented in Jupyter Notebook.
- **Internet connection**: Required for installing necessary dependencies and libraries.


### 2. Required Dataset Download
Please refer to the README.md file in Data/ for downloading GS-Blur dataset and store it properly.

### 3. Python Dependencies--Required Libraries
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
After installing these libraries, you should be able to run the Jupyter Notebook scripts without missing dependencies.

----------------------------------------

## Navigation Instructions
The study follows a comparative experimental design, where different models are trained on varying datasets and their performance is evaluated using PSNR, SSIM, and LPIPS scores. Specifically, three models are trained and tested:
- **a)**	WEDDM trained on a simulated dataset `Simulated_WEDDM.ipynb`
- **b)**	MPRNet trained on the GS-Blur dataset `GSBlur_MPRNet.ipynb`
- **c)**	WEDDM trained on the GS-Blur dataset `AddNoise_GSBlur_10WEDDM.ipynb`&`AddNoise_GSBlur_50WEDDM.ipynb`

All the mentioned Jupyter notebook files can be found in the `Code` folder.

### Steps to Reproduce the Results:

1. **Training the Models:**
   - First, run the following four Jupyter notebooks to obtain the training results for the three models:
     - `Simulated_WEDDM.ipynb`
     - `GSBlur_MPRNet.ipynb`
     - `AddNoise_GSBlur_10WEDDM.ipynb`
     - `AddNoise_GSBlur_50WEDDM.ipynb`

   **Note:**
   - The `AddNoise_GSBlur_50WEDDM.ipynb` was created because running the `AddNoise_GSBlur_10WEDDM.ipynb` model trained for 10 epochs did not meet expectations. Detailed analysis and explanations can be found in the report.
   - The training time for `GSBlur_MPRNet.ipynb` may take several hours, so please consider whether you want to reproduce it yourself.

2. **Visualizing the Results:**
   - After obtaining the training results, you can run the `ResultsVisualization.ipynb` file. The file contains the following pre-computed scores:
     ```python
     psnr_scores = [31.0597, 8.8505, 20.3270, 21.1060]
     ssim_scores = [0.9563, 0.2413, 0.7277, 0.7588]
     lpips_scores = [0.0105, 0.6165, 0.2872, 0.2575]
     ```
     These scores correspond to the evaluation results of the four training outcomes mentioned above. If you have re-run the programs and want to update the PSNR, SSIM, and LPIPS scores, please reset the lists `psnr_scores`, `ssim_scores`, and `lpips_scores` in the order of:
     - `Simulated_WEDDM`
     - `GSBlur_MPRNet`
     - `GSBlur_WEDDM_10Epochs`
     - `GSBlur_WEDDM_50Epochs`

   Running the `ResultsVisualization.ipynb` file will generate comparative graphs of the different models' performances, providing an intuitive understanding of how these models perform on the deblurring task.

3. **Generating Saliency Maps:**
   - The `Appendix.ipynb` file contains code to generate saliency maps, which visualize the important features influencing the model’s predictions. This can be useful for understanding the decision-making process of the models.

----------------------------------------

## Report
[LinZhang_FinalProjectReport](https://github.com/Rising-Stars-by-Sunshine/LinZhang_FinalProject_25Spr_STATS201/blob/29133fbec2447b9ebb26ef1726c0dbb584a69f2a/LinZhang_FinalProjectReport.pdf)

## Poster
[LinZhang_FinalProjectPoster]()


----------------------------------------

## Acknowledgments
I would like to express my gratitude to the following individuals and resources for their invaluable contributions to this project:

- **Professor [Luyao Zhang]**: Thank you for your guidance, insightful feedback, and continuous support throughout this project. Your expertise and encouragement have been instrumental in shaping this work.
- **My Peers**: I am deeply thankful to my classmates for their constructive peer reviews and suggestions, which helped refine and improve the quality of this project.
- **Open-Source Community**: This project benefited greatly from the wealth of knowledge and resources provided by the open-source community. Special thanks to the developers and researchers who shared their code, datasets, and tools publicly.
- **AI Tools**: I also acknowledge the use of AI tools for assisting in code development, debugging, and documentation. These tools significantly enhanced the efficiency and quality of this work.

Your collective support and contributions have been essential to the success of this project. Thank you!



## Statement of Intellectual and Professional Growth
This project has been a transformative learning experience, significantly contributing to both my intellectual and professional development. Below are some key reflections on my growth throughout this journey:

1. **Technical Skills**:
   - I gained hands-on experience with advanced machine learning models, particularly in image deblurring tasks, and deepened my understanding of evaluation metrics such as PSNR, SSIM, and LPIPS.
   - Working with Jupyter Notebooks and Python allowed me to refine my coding practices, improve my debugging skills, and learn efficient ways to manage and visualize data.

2. **Problem-Solving and Critical Thinking**:
   - The challenges encountered during model training and evaluation, such as unexpected results or long training times, taught me the importance of iterative problem-solving and adaptability.
   - Analyzing why certain models underperformed (e.g., the 10-epoch WEDDM model) and devising solutions (e.g., extending training to 50 epochs) enhanced my ability to critically assess and improve my work.

3. **Collaboration and Feedback**:
   - Engaging in peer reviews and incorporating feedback from classmates helped me appreciate the value of diverse perspectives and constructive criticism in refining my project.
   - Collaborating with my professor and peers also improved my communication skills, particularly in explaining technical concepts and justifying methodological choices.

4. **Exposure to Open-Source Tools and AI Technologies**:
   - Leveraging open-source libraries and AI tools not only accelerated my workflow but also exposed me to the collaborative and innovative spirit of the open-source community.
   - This experience inspired me to contribute back to the community by sharing my work and findings.

5. **Professional Growth**:
   - Managing the project from start to finish—from planning and implementation to evaluation and documentation—taught me valuable project management skills, including time management and prioritization.
   - The experience of writing detailed reports and creating visualizations to present my results has strengthened my ability to communicate complex ideas clearly and effectively.

Overall, this project has been a cornerstone of my academic and professional journey, equipping me with the skills, confidence, and mindset to tackle future challenges in machine learning and beyond. I am excited to continue building on this foundation as I progress in my career.
















