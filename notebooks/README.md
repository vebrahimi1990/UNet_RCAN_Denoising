# Parameters
Here is an overview of adjustable parameters within the framework with their meaning and example values. </br>
Use config_2D.yml, train_2D.ipynb, and evaluate_2D.ipynb for the 2D model and config_3D.py, train_3D.ipynb, and evaluate_3D.ipynb for the 3D model. </br>

### data
GT_image_dr: path to high SNR tif stack, "path_to_tif.tif" </br>
lowSNR_image_dr: path to low SNR tif stack, "path_to_tif.tif" </br>
patch_size: patch size in pixels, 256 </br>
n_patches: number of patches per image, 1 </br>
n_channel: number of channels, 0 </br>
threshold: removes empty patches, False or float, e.g. 0.4 </br>
lp: defines the lambda in np.random.poisson, only used if add_noise=True, e.g. 0.5 </br>
normalization_type: if instances are normalized per instance (min max values per instance) or per dataset (global min max values), options are instance/dataset </br>
normalization_range: min and max pixel values after normalization, [0, 1] </br>
add_noise: if Poisson noise should be added to low, False to deactivate it </br>
shuffle: if stack should be shuffled, True/False </br>
augment: if flipping should be added as augmentation, True/False </br>

### data_test
GT_image_dr: "/home/jrahm/denoising/data/2023-11-13_ER_KDEL_Treated/test/high.tif" </br>
lowSNR_image_dr: "/home/jrahm/denoising/data/2023-11-13_ER_KDEL_Treated/test/low.tif" </br>
save_dr: "/home/jrahm/denoising/results/unet_rcan/2023-11-13_ER/lr0.0001_ps256_nf2_l20_er0.05_actleaky_relu_kiglorot_uniform_cv0.01" </br>
patch_size: patch size of test data, if the entire image should be evaluated here, define the image size in px, 600 </br>
n_patches: number of patches per image, 1 </br>
n_channel: number of channels, 0 </br>
threshold: removes empty patches, False or float, e.g. 0.4 </br>
lp: defines the lambda in np.random.poisson, only used if add_noise=True, e.g. 0.5 </br>
normalization_type: if instances are normalized per instance (min max values per instance) or per dataset (global min max values), options are instance/dataset </br>
normalization_range: min and max pixel values after normalization, [0, 1] </br>
add_noise: if Poisson noise should be added to low, False to deactivate it </br>
shuffle: if stack should be shuffled, True/False </br>
augment: if flipping should be added as augmentation, True/False </br>

### model
filters: number of filters in unet, the first filter number is used in rcan, [64, 128, 256] </br>
filters_cab: number of filters in cab, 4 </br>
num_RG: number of residual groups in RCAN, 3 </br>
num_cab: number of channel attention blocks in RCAN, 8 </br>
kernel: kernel size of filters in pixel, 7 </br>
dropout: dropout in U-Net and RCAN network, 0.2 </br>
lr: initial learning rate, 0.0001 </br>
n_epochs: number of epochs, 100 </br>
batch_size: batch size, 10 </br>
activation: activation function, options are leaky_relu and tanh </br>
kernel_initializer: how to initialize the kernel values, options are glorot_uniform, lecun_normal, lecun_uniform, orthogonal, ... try glorot_uniform first </br>
clip_value: gradient clipping, False or float value, e.g. 0.01 </br>
loss_type: custom </br>
norm_factor: the loss is normalized by patch_size x patch_size x norm_factor, 1 </br>
l2_regularization: amount of L2 regularization, 0=deactivated or float value, e.g. 0.1 </br>
edge_regularization: amount of edge image in loss function, 0=deactivated or float value, e.g. 0.05 </br>
save_dr: path to save model, "/path_to_dir/model.h5" </br>
save_config: save directory, "/path_to_dir" </br>

### callbacks
patience_stop: number of epochs to terminate training if validation loss has not improved, 6 </br>
factor_lr: factor to reduce learning rate after patience_lr epochs, 0.2 </br>
patience_lr: number of epochs after which the learning rate is reduced by factor_lr, 4 </br>
num_cab: number of channel attention blocks in RCAN, 8 </br>
kernel: kernel size of filters in pixel, 7 </br>
dropout: dropout in U-Net and RCAN network, 0.2 </br>
lr: initial learning rate, 0.0001 </br>
n_epochs: number of epochs, 100 </br>
batch_size: batch size, 10 </br>
val_split: validation split, 0.1 </br>
save_dr: path to save model, "/path_to_dir/model.h5" </br>