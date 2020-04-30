import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_profile(x):
    plt.ylim((0., 2.))
    plt.plot(x, linewidth=1.4)
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    input_file = '/media/julia/Data/datasets/lufpro/real/simple_preprocess3/Razieh05.09_1Tow_Setup8_6,0m_min_3.csv'
    df = pd.read_csv(input_file, header=None)
    x_data = df.loc[:, 5:804].to_numpy()
    y_data = df.loc[:, 805:].to_numpy()
    del df
    plot_profile(x_data[30])
    plot_profile(y_data[30])
