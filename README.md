# UNet_RCAN_Denoising
UNet-RCAN is a two-step prediction algorithm for supervised denosing of super-resolution imaging data built in Tensorflow 2.7.0 framework.

# Dependencies
```pip install -r requirements.txt```

# Notebooks
Notebooks are in the notebook folder. 

# Training
```!git clone https://github.com/vebrahimi1990/UNet_RCAN_Denoising.git```

For training, add the directory to your training dataset and a directory to save the model to the configuration file ```(config_(2D/3D).py)```, then train the model using ```train_(2D/3D).py```. 

# Evaluation
For evaluation, add the directory to your test dataset and a directory to the saved model to the configuration file ```(config_(2D/3D).py)```, then test the model using  ```evaluate_(2D/3D).py```. 

# Architecture
![plot](https://github.com/vebrahimi1990/UNet_RCAN_Denoising/blob/master/image%20files/Architecture.png)

# Results
