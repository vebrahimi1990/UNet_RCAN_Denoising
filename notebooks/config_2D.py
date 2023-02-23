from collections import OrderedDict
CFG = OrderedDict()

CFG = {
    "data": {
        "GT_image_dr": r"\.tif",
        "lowSNR_image_dr": r"\.tif",
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
        "GT_image_dr": r"\.tif",
        "lowSNR_image_dr": r"\.tif",
        "save_dr": r"",
        "patch_size": 1024,
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
        "save_dr": r"\.h5",
        "save_config": r""
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
        "save_dr": r"\.h5"
    }
}
