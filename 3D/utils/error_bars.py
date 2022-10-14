import matplotlib.pyplot as plt
import numpy as np

from args import args

plt.switch_backend('agg')


def errorfill(x, y, yerr, color=None, alpha_fill=0.3, ax=None):
    ax = ax if ax is not None else plt.gca()
    if color is None:
        color = ax._get_lines.color_cycle.next()
    if np.isscalar(yerr) or len(yerr) == len(y):
        ymin = y - yerr
        ymax = y + yerr
    elif len(yerr) == 2:
        ymin, ymax = yerr
    ax.plot(x, y, color=color)
    ax.fill_between(x, ymax, ymin, color=color, alpha=alpha_fill)


def error_bar(actual, pred, epoch, layer):
    actual = actual
    actual = actual.reshape(64, 64).transpose()
    pred_mean = np.mean(pred, axis=0).transpose()
    pred_mean = pred_mean.reshape(64, 64).transpose()
    pred_diag = np.diag(pred_mean)
    actdiag = np.diag(actual)
    pred_std = np.std(pred, axis=0).transpose()
    pred_std = pred_std.reshape(64, 64)
    std_diag = np.diag(pred_std)
    std_val = np.std(actdiag)
    x = np.linspace(0, 64, 64)
    errorfill(x, pred_diag, 2 * std_diag, 'b')
    plt.plot(x, actdiag, 'g')
    plt.savefig('./{}/diag_error_{}_layer{}.pdf'.format(args.results_path, epoch, layer), bbox_inches='tight')
    plt.close()


def train_test_error(nll_train, nll_test, epoch):
    plt.figure()
    plt.plot(nll_test, label="Test: {:.3f}".format(np.mean(nll_test[-5:])))
    plt.xlabel('Epoch')
    plt.ylabel(r'NLL')
    plt.legend(loc='lower right')
    plt.savefig(f"{args.results_path}/Test_nll.pdf", dpi=600)
    plt.close()
    np.savetxt(f"./{args.results_path}/Test_nll.txt", nll_test)


def plot_std(samples, epoch):
    plt.imshow(samples, cmap='jet', origin='lower', interpolation='bilinear')
    plt.colorbar()
    plt.tight_layout(pad=0.05, w_pad=0.05, h_pad=0.05)
    plt.savefig(f'{args.results_path}/std_%d.pdf' % epoch, dpi=300, bbox_inches='tight')
    plt.close()
