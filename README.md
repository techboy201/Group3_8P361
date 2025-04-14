# 8P361 Project AI for MIA - Group 3
*Does a convolutional neural network pick up on artificially added confounding factors?*

This repository contains python code used to generate results for group 3's submission of the assignments and main project of 8P361 (Project AI for MIA). 

### Running code
An external dataset based on the PatchCamelyon dataset is required to run the code, which can be downloaded [here](https://tuenl-my.sharepoint.com/:u:/g/personal/q_m_liu_student_tue_nl/EUf4_iiE7wtPiJVnpjbiQocBujRptQYIN9IK26JzFiHVtA?e=XQGNCz). This file should be unpacked in a way that produces a `/datasets/` folder in the root of the repository.
Two extra python packages were used: cv2 and seaborn. These need to be installed along with the base packages of this course to run the code.
The code in the `/main_project/` folder was used for the final results. First the python files `Confounding factors with coordinates.py` alongside and `ccn_fraction_confounded.py` in order to get models trained on different amounts of augmented data. The models can then be used by `Grad_cam.py' and 'results_analysis.py` to gather the results.  

## References

The following resources were used as a foundation for the Grad-CAM code and GMI analysis:

- Selvaraju et al. (2017). Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization. [arXiv:1610.02391](https://arxiv.org/abs/1610.02391)
- Matplotlib Visualization: [https://matplotlib.org](https://matplotlib.org)
- Pillow (PIL): [https://pillow.readthedocs.io](https://pillow.readthedocs.io)
- Stack Overflow:[https://stackoverflow.com/questions/55266249/create-a-mixed-data-generator-images-csv-in-keras](https://stackoverflow.com/questions/55266249/create-a-mixed-data-generator-images-csv-in-keras) and [https://stackoverflow.com/questions/66182884/how-to-implement-grad-cam-on-a-trained-network](https://stackoverflow.com/questions/66182884/how-to-implement-grad-cam-on-a-trained-network)
- Keras examples: [https://keras.io/examples/vision/grad_cam/](https://keras.io/examples/vision/grad_cam/) 

