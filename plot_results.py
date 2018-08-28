import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt
import json
import numpy as np
import pickle


def plot_results(hist_file, save_dir, mode):
    if mode == 'pi':
        hist_pi_file = hist_file
        with open(hist_pi_file, 'rb') as fp:
            hist = pickle.load(fp)

        ##- Setting up a plot for loss and training acc. graphs
        # --------------------------------------------------------
        plt.plot(hist['loss'], linewidth=2, color='b', label='Train')
        plt.plot(hist['val_loss'], linewidth=2, color='r', label='Valida.')

        plt.grid()
        # ~ plt.grid(linestyle='dotted')
        plt.grid(color='black', linestyle='--', linewidth=1)
        plt.ylabel('Loss', fontsize=18)
        plt.xlabel('Epoch', fontsize=18)
        plt.xticks(fontsize=12, rotation=0)
        plt.yticks(fontsize=12, rotation=0)
        plt.title('Loss')
        plt.legend(shadow=False, fancybox=False)
        plt.tight_layout()
        # ~ plt.show()
        plt.savefig(save_dir + 'loss_test.png')
        plt.close()

        plt.plot(hist['acc'], linewidth=3, color='b', label='Train')
        plt.plot(hist['val_acc'], linewidth=3, color='r', label='Valida.')

        plt.grid()
        plt.grid(color='black', linestyle='--', linewidth=1)
        plt.ylabel('Accuracy', fontsize=18)
        plt.xlabel('Epoch', fontsize=18)
        plt.xticks(fontsize=12, rotation=0)
        plt.yticks(fontsize=12, rotation=0)
        plt.title('Accuracy')
        plt.legend(shadow=False, fancybox=False, loc='lower right')
        plt.tight_layout()
        plt.savefig(save_dir + 'acc_test.png')
        # ~ plt.close()

    elif mode == 'json':
        hist_json_file = hist_file
        with open(hist_json_file, "r") as f:
            hist = json.load(f)

        ##- Setting up a plot for loss and training acc. graphs
        # --------------------------------------------------------

        plt.plot(np.squeeze(hist['loss']), linewidth=2, color='b', label='Train')
        plt.plot(np.squeeze(hist['val_loss']), linewidth=2, color='r', label='Valida.')

        plt.grid()
        # ~ plt.grid(linestyle='dotted')
        plt.grid(color='black', linestyle='--', linewidth=1)
        plt.ylabel('Loss', fontsize=18)
        plt.xlabel('Epoch', fontsize=18)
        plt.xticks(fontsize=12, rotation=0)
        plt.yticks(fontsize=12, rotation=0)
        plt.title('Loss')
        plt.legend(shadow=False, fancybox=False)
        plt.tight_layout()
        # ~ plt.show()
        plt.savefig(save_dir + 'loss_test.png')
        plt.close()

        plt.plot(np.squeeze(hist['acc']), linewidth=3, color='b', label='tr_bp_acc')
        plt.plot(np.squeeze(hist['val_acc']), linewidth=3, color='r', label='te_bp_acc')
        plt.plot(np.squeeze(hist['w12_acc']), linewidth=3, color='g', label='te_w12_acc')

        plt.grid()
        plt.grid(color='black', linestyle='--', linewidth=1)
        plt.ylabel('Accuracy', fontsize=18)
        plt.xlabel('Epoch', fontsize=18)
        plt.xticks(fontsize=12, rotation=0)
        plt.yticks(fontsize=12, rotation=0)
        plt.title('Accuracy')
        plt.legend(shadow=False, fancybox=False, loc='lower right')
        plt.tight_layout()
        plt.savefig(save_dir + 'acc_test.png')
        # ~ plt.close()