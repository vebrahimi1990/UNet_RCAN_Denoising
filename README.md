# UNet_RCAN_Denoising
[![paper](https://img.shields.io/badge/bioRxiv-Paper-brightgreen)](https://www.biorxiv.org/content/biorxiv/early/2023/01/27/2023.01.26.525571.full.pdf)

UNet-RCAN is a two-step prediction algorithm for supervised denosing of fast stimulated emission depletion microscopy (STED) data, built in Tensorflow 2.7.0 framework.

# Dependencies
```pip install -r requirements.txt```

# Notebooks
Notebooks are in the notebook folder. 

# Training
```git clone https://github.com/vebrahimi1990/UNet_RCAN_Denoising.git```

For training, add the directory to your training dataset and a directory to save the model to the configuration file ```(config_(2D/3D).py)```, then train the model using ```train_(2D/3D).py```. 

# Evaluation
For evaluation, add the directory to your test dataset and a directory to the saved model to the configuration file ```(config_(2D/3D).py)```, then test the model using  ```evaluate_(2D/3D).py```. 

# Architecture
![plot](https://github.com/vebrahimi1990/UNet_RCAN_Denoising/blob/master/image%20files/Architecture.png)

# Results
![plot](https://github.com/vebrahimi1990/UNet_RCAN_Denoising/blob/master/image%20files/Results.png)

# Contact
Should you have any question, please contact vebrahimi1369@gmail.com. 
