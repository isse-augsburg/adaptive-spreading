from pathlib import Path


DATA = 'adaptive_spreading_preprocessed_data'
NN_WEIGHTS = Path('weights/')
NN_MODELS = Path('models/')
NN_LOGS = 'logs/'
NN_TENSORBOARD = 'runs/'
RF_MODELS = Path('rf_models/')
RF_LOGS = 'logs/'
NE_WEIGHTS = Path('ne_weights/')
NE_MODELS = Path('ne_models/')
NE_LOGS = 'process_control_model/logs/'
NE_PARAMS = 'process_control_model/params/'
NE_TENSORBOARD = 'ne_runs/'
# NN_BACKEND = 'models/300_300_300_relu_lr=0.0001' \
#              '_batch_size=64_split_data_respecting_files_simpleprep3_param=1111.pth'
NN_BACKEND = 'models/100_100_100_relu_lr=0.0001' \
             '_batch_size=64_split_data_respecting_files_new_eval_param=1111.pth'
RF_BACKEND = 'rf_models/est_50_maxdepth_12_min_split_150_max_feats_200_' \
             'simpleprep3_random_forest_regressor.joblib'
