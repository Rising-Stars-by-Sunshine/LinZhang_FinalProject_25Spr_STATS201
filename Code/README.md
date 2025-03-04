## Steps to Reproduce the Results:

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
