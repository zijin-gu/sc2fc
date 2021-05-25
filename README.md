# sc2fc

This is an adaptation of the code by Tabinda Sarwar (https://github.com/sarwart/mapping_SC_FC).
Trained on 340 unrelated HCP subjects and tested on 80 unrelated HCP subjects in fs86 atlas.

## requirements
Tenserflow (my current version is 1.1.0, some functions may subject to change based on different versions).

## usage
Please update the path to your test data, the path to save your predicted data and the GPU ID in `reload.py`.

Currently the best model is `model_gam1_lam4.ckpt-20000` which has hyperparameters `gamma=0.2` and `lambda=0.01`.

If you would like to train your own model, please directly refer to the original github repo.
