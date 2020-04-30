import numpy as np
import torch
import matplotlib.pyplot as plt


def plot_samples(model_name, sensor_dim, data, meta=None):
    num_samples = 10
    ixs = np.random.randint(len(data[0]), size=num_samples)
    preds = data[0].view(-1, sensor_dim)
    targets = data[1].view(-1, sensor_dim)
    print(ixs)
    plot_i = 1
    for i in ixs:
        if meta:
            plt.subplot(num_samples, 1, plot_i,
                        title=f'{meta[0][i]}, line {int(meta[1][i].item())}')
            plt.subplots_adjust(hspace=.5)
        else:
            plt.subplot(num_samples, 1, plot_i)
        plt.plot(targets[i], linewidth=0.4)
        plt.plot(preds[i], linewidth=0.4)
        plot_i += 1
    plt.legend(['target', 'prediction'])
    plt.suptitle(model_name)
    plt.show()


def plot_samples_np_wrapper(preds, targets, sensor_dim, name='Random forest', meta=None):
    if meta is not None:
        meta = meta[0], torch.tensor(meta[1])
    plot_samples(name, sensor_dim, (torch.tensor(preds), torch.tensor(targets)), meta=meta)
