CFG = {
    "data": {
        "GT_image_dr": r"\train\Average.tif",
        "lowSNR_image_dr": r"\train\1frame.tif",
        "patch_size": 256,
        "n_patches": 16,
        "n_channel": 0,
        "threshold": 0.4,
        "lp": 0.5,
        "add_noise": False,
        "shuffle": True,
        "augment": False
    },
    "data_test": {
        "GT_image_dr": r"\test\Average.tif",
        "lowSNR_image_dr": r"\test\1frame.tif",
        "save_dr": r"D:\Projects\Denoising-STED",
        "patch_size": 2048,
        "n_patches": 1,
        "n_channel": 0,
        "threshold": 0.0,
        "lp": 0.5,
        "add_noise": False,
        "shuffle": False,
        "augment": False
    },
    "model": {
        "filters": [64, 128, 256],
        "filters_cab": 4,
        "num_RG": 3,
        "num_cab": 8,
        "kernel": 3,
        "dropout": 0.2,
        "lr": 0.0001,
        "n_epochs": 200,
        "batch_size": 1,
        "save_dr": r"D:\Projects\Denoising-STED\model.h5"
    },
    "callbacks": {
        "patience_stop": 20,
        "factor_lr": 0.2,
        "patience_lr": 5,
        "num_cab": 8,
        "kernel": 3,
        "dropout": 0.2,
        "lr": 0.0001,
        "n_epochs": 200,
        "batch_size": 1,
        "save_dr": r"D:\Projects\Denoising-STED\model.h5"
    }
}
