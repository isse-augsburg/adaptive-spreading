from pathlib import Path
import pandas as pd
import numpy as np
from preprocessing.simple_preprocessing import subproc
from process_model.model_comparison import plot_samples

if __name__ == '__main__':
    filename = Path('/media/julia/Data/datasets/lufpro/real/Razieh11.09_1Tow_Setup10_8,0m_min_3.csv')
    #preprocessed_data = subproc(filename)
    sensor_dim = 800
    params_dim = 17

    data = pd.read_csv(filename)
    data.columns = range(data.shape[1])
    num = data._get_numeric_data()
    num[num < -100] = np.nan
    xmean_val = np.nanmean(num.loc[:, params_dim:sensor_dim + params_dim - 1])
    ymean_val = np.nanmean(num.loc[:, sensor_dim + params_dim:])
    x = num.values
    params_dim = x.shape[-1] - 2 * sensor_dim

    y = x[:, (sensor_dim + params_dim):] #- ymean_val
    x = x[:, params_dim:(sensor_dim + params_dim)] #- xmean_val

    # x[x < xmean_val-0.5] = np.nan
    # x[x > xmean_val+0.5] = np.nan
    # y[y < ymean_val-0.5] = np.nan
    # y[y > ymean_val+0.5] = np.nan
    select_data = np.random.choice(x.shape[0], 8, replace=False)
    # preprocessed_data[select_data, 5:805]
    plot_samples([x[select_data], y[select_data]], legend=['raw', 'preprocessed'],
                 title='Pre-processing', plot_edges=False)
