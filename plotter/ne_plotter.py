from _tkinter import TclError
import matplotlib.pyplot as plt

from preprocessing.tape_detection import get_tape_edges
from process_control_model.ne_utils import create_target


def render(tape_in, tape_out, bar_positions, target_width, target_height, sensor_dim=800, setup_dim=5):
    target = create_target(target_width, target_height, dim=1, sensor_dim=sensor_dim).squeeze()
    fig, axs = plt.subplots(3)
    fig.set_size_inches(16, 14)
    plt.show(block=False)
    raising_edges, falling_edges = get_tape_edges(tape_out)
    for i, (start, row, e1, e2, action) in enumerate(zip(tape_in, tape_out, raising_edges, falling_edges,
                                                         bar_positions)):
        try:
            axs[0].clear()
            axs[0].plot(start, '-b', label='start values')
            axs[0].legend()
            axs[0].set_ylim([-0.3, 0.6])
            axs[1].clear()
            axs[1].plot(target, color='red', label='target values')
            axs[1].plot(row, '-bo', markerfacecolor='r',
                        markevery=[e1, e2], label='end values')
            axs[1].legend()
            axs[1].set_ylim([-0.3, 0.6])
            axs[2].clear()
            axs[2].plot(action, '-ko', markerfacecolor='r', markevery=[i for i in range(setup_dim)],
                        label='bar position')
            axs[2].legend()
            axs[2].set_ylim([10, 40])
            fig.canvas.draw()
            plt.pause(0.001)
        except (KeyboardInterrupt, TclError):
            break
    plt.close(fig)
