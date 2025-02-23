# LinZhang_FinalProject_25Spr_STATS201

# Wavelet-Enhanced Deblurring Diffusion Model (WEDDM) on GS-Blur Dataset

## Background and Motivation
Image deblurring is a crucial task in computer vision with wide-ranging applications in fields such as surveillance, autonomous driving, and remote sensing. Traditional methods, although effective under controlled conditions, often struggle in real-world scenarios where noise and blur conditions vary greatly. Most existing deep learning-based deblurring models are trained in low signal-to-noise ratio (SNR) environments, where the models perform well in simulated, noise-free conditions but fail to generalize to images with high noise and complex real-world blur. This research seeks to address this gap by applying the Wavelet-Enhanced Deblurring Diffusion Model (WEDDM), a state-of-the-art blind deblurring algorithm, to the GS-Blur dataset, which simulates more realistic near-field blur conditions. The goal is to evaluate whether WEDDM can effectively handle real-world, high-noise blur scenarios and improve performance in everyday image deblurring tasks.

## Research Question
The central research question of this study is: How does the Wavelet-Enhanced Deblurring Diffusion Model (WEDDM) perform when trained on the GS-Blur dataset, particularly in restoring near-field images with complex blur patterns and varying noise levels?

## Application Scenarios
The outcomes of this research have broad applications across various domains that require high-quality image restoration, especially in noisy and blurred conditions. Potential applications include promoting autonomous driving technologies, where deblurring can improve vehicle perception in challenging weather or lighting conditions. Aside from that, it could also strengthen security and surveillance, where improving the clarity of video footage can enhance object recognition and tracking accuracy.

## Methodology
This study will focus on leveraging the GS-Blur dataset for training and evaluating the Wavelet-Enhanced Diffusion Model (WEDDM), a state-of-the-art deblurring algorithm. The methodology will focus on the model’s ability to generalize and enhance performance in near-field (ordinary) image deblurring tasks, addressing the gap between existing low-SNR training environments and real-world high-SNR conditions.

### Machine Learning for Explanation:
For the explanation of the model’s behavior, we will utilize interpretability techniques to gain insights into how the WEDDM functions during image restoration. Specifically, we will use saliency maps and attention visualizations to investigate which parts of the input image contribute most to the deblurring process.

### Machine Learning for Prediction:
To predict the effectiveness of deblurring under varying levels of noise, we will employ supervised machine learning techniques. Specifically, a neural network model will be trained to predict the quality of the restored image, utilizing features such as blur kernel size, noise levels, and pixel intensities.

### Causal Inference using Machine Learning:
To explore the impact of synthetic data on model performance, we will apply a Regression Discontinuity (RD) design to assess the causal effect of training the WEDDM on GS-Blur dataset vs. traditional datasets on the model’s generalization ability.

## Results
The expected results of this study include improved image deblurring performance under high noise and complex blur conditions, as indicated by higher PSNR, SSIM, and LPIPS scores. Additionally, WEDDM is expected to show better generalization when trained on the GS-Blur dataset compared to traditional deblurring models.

## Intellectual Merit and Practical Impacts
This research will contribute to the field of image restoration by testing the WEDDM in new contexts---specifically, near-field image deblurring---and exploring its performance in more realistic conditions, such as higher noise levels and dynamic blur. By training and evaluating WEDDM on the GS-Blur dataset, this study will help bridge the gap between synthetic training data and real-world image restoration tasks, pushing the boundaries of deblurring models beyond traditional applications.
