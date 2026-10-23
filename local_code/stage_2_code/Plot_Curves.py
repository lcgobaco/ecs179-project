'''
Plot the learning convergence curves recorded by Method_MLP
'''

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_history(history, title, file_path):
    # training loss and train/test accuracy against the epoch number
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(10, 4))
    ax_loss.plot(history['epoch'], history['train_loss'], marker='o', markersize=3)
    ax_loss.set_xlabel('Epoch')
    ax_loss.set_ylabel('Training loss')
    ax_loss.set_title('Training loss')
    ax_loss.grid(alpha=0.3)
    ax_acc.plot(history['epoch'], history['train_acc'], marker='o', markersize=3, label='train')
    ax_acc.plot(history['epoch'], history['test_acc'], marker='o', markersize=3, label='test')
    ax_acc.set_xlabel('Epoch')
    ax_acc.set_ylabel('Accuracy')
    ax_acc.set_title('Accuracy')
    ax_acc.legend()
    ax_acc.grid(alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(file_path, dpi=150)
    plt.close(fig)


def plot_comparison(histories, key, ylabel, title, file_path):
    # one curve per experiment for a single recorded quantity, e.g. 'train_loss' or 'test_acc'
    fig, ax = plt.subplots(figsize=(6, 4))
    for name, history in histories.items():
        ax.plot(history['epoch'], history[key], label=name)
    ax.set_xlabel('Epoch')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(file_path, dpi=150)
    plt.close(fig)
