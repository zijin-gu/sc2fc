# sc2fc

This is an adaptation of the code by Tabinda Sarwar (https://github.com/sarwart/mapping_SC_FC).
Trained on HCP data in fs86 atlas.

## requirements
Tenserflow (my current version is 1.1.0, some functions may subject to change based on different versions).

## usage
Please update the path to your test data and the path to save your predicted data in `reload.py`.

Currently the best model is `model_gam1_lam4.ckpt-20000` which has hyperparameters `gamma=0.2` and `lambda=0.01`.
