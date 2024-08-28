import random
import glob
import numpy as np
from tifffile import imread
random.seed(42)


def data_generator_2D(data_config):
    GT_image_dr = data_config['GT_image_dr']
    lowSNR_image_dr = data_config['lowSNR_image_dr']
    patch_size = data_config['patch_size']
    n_patches = data_config['n_patches']
    n_channel = data_config['n_channel']
    threshold = data_config['threshold']
    lp = data_config['lp']
    augment = data_config['augment']
    shuffle = data_config['shuffle']
    add_noise = data_config['add_noise']
    normalization_type = data_config['normalization_type']
    normalization_range = data_config['normalization_range']

    gt = imread(GT_image_dr).astype(np.float64)
    low = imread(lowSNR_image_dr).astype(np.float64)
    if len(gt.shape) == 3:
        gt = np.reshape(gt, (gt.shape[0], 1, gt.shape[1], gt.shape[2]))
        low = np.reshape(low, (low.shape[0], 1, low.shape[1], low.shape[2]))
    if len(gt.shape) == 2:
        gt = np.reshape(gt, (1, 1, gt.shape[0], gt.shape[1]))
        low = np.reshape(low, (1, 1, low.shape[0], low.shape[1]))
    m = gt.shape[0]
    img_size = gt.shape[2]
    
    if normalization_type == 'instance':
        gt = gt / (gt.max(axis=(-1, -2))).reshape((gt.shape[0], gt.shape[1], 1, 1))
        low = low / (low.max(axis=(-1, -2))).reshape((low.shape[0], low.shape[1], 1, 1))
    elif normalization_type == 'dataset':
        gt = gt / gt.max()
        low = low / low.max()
    
    low = np.nan_to_num(low, nan=0.0)
    gt = np.nan_to_num(gt, nan=0.0)

    if add_noise:
        for i in range(len(gt)):
            low[i] = np.random.poisson(gt[i] / lp, size=gt[i].shape)

    x = np.empty((m * n_patches * n_patches, patch_size, patch_size, 1), dtype=np.float64)
    y = np.empty((m * n_patches * n_patches, patch_size, patch_size, 1), dtype=np.float64)

    rr = np.floor(np.linspace(0, img_size - patch_size, n_patches)).astype(np.int32)
    cc = np.floor(np.linspace(0, gt.shape[3] - patch_size, n_patches)).astype(np.int32)

    count = 0
    for l in range(m):
        for j in range(n_patches):
            for k in range(n_patches):
                x[count, :, :, 0] = low[l, n_channel, rr[j]:rr[j] + patch_size, cc[k]:cc[k] + patch_size]
                y[count, :, :, 0] = gt[l, n_channel, rr[j]:rr[j] + patch_size, cc[k]:cc[k] + patch_size]
                count = count + 1

    if augment:
        count = x.shape[0]
        xx = np.zeros((4 * count, patch_size, patch_size, 1), dtype=np.float64)
        yy = np.zeros((4 * count, patch_size, patch_size, 1), dtype=np.float64)

        xx[0:count, :, :, :] = x
        xx[count:2 * count, :, :, :] = np.flip(x, axis=1)
        xx[2 * count:3 * count, :, :, :] = np.flip(x, axis=2)
        xx[3 * count:4 * count, :, :, :] = np.flip(x, axis=(1, 2))

        yy[0:count, :, :, :] = y
        yy[count:2 * count, :, :, :] = np.flip(y, axis=1)
        yy[2 * count:3 * count, :, :, :] = np.flip(y, axis=2)
        yy[3 * count:4 * count, :, :, :] = np.flip(y, axis=(1, 2))
    else:
        xx = x
        yy = y


    if threshold != False:
        norm_x = np.linalg.norm(yy, axis=(1, 2))
        norm_x = norm_x / norm_x.max()
        norm_x = np.nan_to_num(norm_x, nan=0.0)
        ind_norm = np.where(norm_x > threshold)[0]

        xxx = np.empty((len(ind_norm), xx.shape[1], xx.shape[2], xx.shape[3]))
        yyy = np.empty((len(ind_norm), xx.shape[1], xx.shape[2], xx.shape[3]))

        for i in range(len(ind_norm)):
            xxx[i] = xx[ind_norm[i]]
            yyy[i] = yy[ind_norm[i]]
    else:
        xxx = xx
        yyy = yy
   

    xxs = np.empty(xxx.shape, dtype=np.float64)
    yys = np.empty(yyy.shape, dtype=np.float64)

    if shuffle:
        aa = np.linspace(0, len(xxx) - 1, len(xxx))
        random.shuffle(aa)
        aa = aa.astype(int)
        for i in range(len(xxx)):
            xxs[i] = xxx[aa[i]]
            yys[i] = yyy[aa[i]]
    else:
        xxs = xxx
        yys = yyy

    xxs[xxs < 0] = 0
    
    if normalization_type == 'instance':
        xxs = xxs / (xxs.max(axis=(1, 2))).reshape((xxs.shape[0], 1, 1, 1))
        yys = yys / (yys.max(axis=(1, 2))).reshape((yys.shape[0], 1, 1, 1))
    elif normalization_type == 'dataset':
        xxs = xxs / xxs.max()
        yys = yys / yys.max()

    x_train = xxs
    y_train = yys
    
    x_train = np.nan_to_num(x_train, nan=0.0)
    y_train = np.nan_to_num(y_train, nan=0.0)
    
    
    if normalization_range != [0, 1]:
        min, max = normalization_range[0], normalization_range[1]
        x_train = min + (max - min) / (1 - 0) * (x_train - 0)
        y_train = min + (max - min) / (1 - 0) * (y_train - 0)
    
    print('Dataset shape:', x_train.shape)
    return x_train, y_train


def stack_generator_3D(GT_dr, low_dr, fr_start, fr_end):
    path_gt = GT_dr + '/*.tif'
    path_low = low_dr + '/*.tif'
    image_gt = imread(sorted(glob.glob(path_gt))).astype(np.float32)
    image_low = imread(sorted(glob.glob(path_low))).astype(np.float32)

    if len(image_gt.shape) == 3:
        image_gt = np.reshape(image_gt, (image_gt.shape[0], 1, 1, image_gt.shape[1], image_gt.shape[2]))
        image_low = np.reshape(image_low,
                               (image_low.shape[0], 1, 1, image_low.shape[1], image_low.shape[2]))

    if len(image_gt.shape) == 4:
        image_gt = np.reshape(image_gt, (image_gt.shape[0], image_gt.shape[1], 1, image_gt.shape[2], image_gt.shape[3]))
        image_low = np.reshape(image_low,
                               (image_low.shape[0], image_low.shape[1], 1, image_low.shape[2], image_low.shape[3]))


    for i in range(len(image_gt)):
        for j in range(image_gt.shape[2]):
            if image_gt[i, :, j, :, :].max() > 0:
                image_gt[i, :, j, :, :] = image_gt[i, :, j, :, :] / image_gt[i, :, j, :, :].max()
            if image_low[i, :, j, :, :].max() > 0:
                image_low[i, :, j, :, :] = image_low[i, :, j, :, :] / image_low[i, :, j, :, :].max()

    crop_gt = image_gt[:, fr_start:fr_end, :, :, :]
    crop_low = image_low[:, fr_start:fr_end, :, :, :]
    crop_gt = np.moveaxis(crop_gt, 1, -1)
    crop_low = np.moveaxis(crop_low, 1, -1)
    crop_gt = np.moveaxis(crop_gt, 1, -1)
    crop_low = np.moveaxis(crop_low, 1, -1)
    return crop_gt, crop_low


def data_generator_3D(data_config):
    GT_dr = data_config['GT_image_dr']
    low_dr = data_config['lowSNR_image_dr']
    patch_size = data_config['patch_size']
    n_patches = data_config['n_patches']
    n_channel = data_config['n_channel']
    threshold = data_config['threshold']
    fr_start = data_config['fr_start']
    fr_end = data_config['fr_end']
    lp = data_config['lp']
    augment = data_config['augment']
    shuffle = data_config['shuffle']
    add_noise = data_config['add_noise']

    gt, low = stack_generator_3D(GT_dr, low_dr, fr_start, fr_end)

    m = gt.shape[0]
    img_size = gt.shape[2]
    z_depth = fr_end - fr_start

    x = np.empty((m * n_patches * n_patches, patch_size, patch_size, z_depth, 1), dtype=np.float32)
    y = np.empty((m * n_patches * n_patches, patch_size, patch_size, z_depth, 1), dtype=np.float32)

    if n_patches == 1:
        rr = [0]
        cc = [0]
    else:
        rr = np.floor(np.linspace(0, img_size - patch_size, n_patches))
        cc = rr
        rr = rr.astype(np.int32)
        cc = cc.astype(np.int32)

    count = 0

    for l in range(m):
        for j in range(n_patches):
            for k in range(n_patches):
                x[count, :, :, :, 0] = low[l, rr[j]:rr[j] + patch_size, cc[k]:cc[k] + patch_size, :, n_channel]
                y[count, :, :, :, 0] = gt[l, rr[j]:rr[j] + patch_size, cc[k]:cc[k] + patch_size, :, n_channel]
                count = count + 1

    # Generating a noisy dataset by adding Poisson noise
    if add_noise:
        for i in range(len(x)):
            x[i] = np.random.poisson((y[i]) / lp, size=y[i].shape)
            # for k in range(x.shape[3]):
            #     pixel_x = np.random.randint(0, patch_size, [2, 10])
            #     pixel_y = np.random.randint(0, patch_size, [2, 10])
            #     x[i, pixel_x[1, :], pixel_y[1, :], k, :] = 0
            #     x[i, pixel_x[0, :], pixel_y[0, :], k, :] = 1

    if augment:
        count = x.shape[0]
        xx = np.zeros((4 * count, patch_size, patch_size, z_depth, 1), dtype=np.float32)
        yy = np.zeros((4 * count, patch_size, patch_size, z_depth, 1), dtype=np.float32)

        xx[0:count, :, :, :, :] = x
        xx[count:2 * count, :, :, :, :] = np.flip(x, axis=1)
        xx[2 * count:3 * count, :, :, :, :] = np.flip(x, axis=2)
        xx[3 * count:4 * count, :, :, :, :] = np.flip(x, axis=(1, 2))

        yy[0:count, :, :, :, :] = y
        yy[count:2 * count, :, :, :, :] = np.flip(y, axis=1)
        yy[2 * count:3 * count, :, :, :, :] = np.flip(y, axis=2)
        yy[3 * count:4 * count, :, :, :, :] = np.flip(y, axis=(1, 2))
    else:
        xx = x
        yy = y

    xx = xx / xx.max()
    yy = yy / yy.max()

    norm_x = np.linalg.norm(np.max(xx, axis=3), axis=(1, 2))
    ind_norm = np.where(norm_x >= threshold)[0]
    print(len(ind_norm))

    xxx = np.empty((len(ind_norm), xx.shape[1], xx.shape[2], xx.shape[3], xx.shape[4]))
    yyy = np.empty((len(ind_norm), xx.shape[1], xx.shape[2], xx.shape[3], xx.shape[4]))

    for i in range(len(ind_norm)):
        xxx[i] = xx[ind_norm[i]]
        yyy[i] = yy[ind_norm[i]]

    aa = np.linspace(0, len(xxx) - 1, len(xxx))
    random.shuffle(aa)
    aa = aa.astype(int)

    xxs = np.empty(xxx.shape, dtype=np.float32)
    yys = np.empty(yyy.shape, dtype=np.float32)

    if shuffle:
        for i in range(len(xxx)):
            xxs[i] = xxx[aa[i]]
            yys[i] = yyy[aa[i]]
    else:
        xxs = xxx
        yys = yyy

    x_train = xxs
    y_train = yys

    print('Dataset shape:', x_train.shape)
    return x_train, y_train
