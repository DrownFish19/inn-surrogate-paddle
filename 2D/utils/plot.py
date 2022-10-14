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


def error_bar(actual, pred, epoch):
    actual = actual
    actual = actual.reshape(64, 64)
    pred = pred

    pred_mean = np.mean(pred, axis=0)
    print('pred_mean', pred_mean.shape, flush=True)
    print(actual.shape, flush=True)
    print(pred.shape, flush=True)

    pred_mean = pred_mean.reshape(64, 64)
    pred_diag = np.diag(pred_mean)

    actdiag = np.diag(actual)
    print(actdiag.shape, flush=True)

    pred_std = np.std(pred, axis=0)
    pred_std = pred_std.reshape(64, 64)
    print('std', pred_std.shape, flush=True)
    std_diag = np.diag(pred_std)

    std_val = np.std(actdiag)
    x = np.linspace(0, 64, 64)
    y_sin = np.sin(x)
    y_cos = np.cos(x)
    errorfill(x, pred_diag, 2 * std_diag, 'b')
    plt.plot(x, actdiag, 'g')
    plt.savefig(f'{args.results_path}/diag_error_{epoch}.pdf', bbox_inches='tight')
    plt.close()


def train_test_error(nll_train, nll_test, epoch):
    plt.figure()
    plt.plot(nll_test, label="Test: {:.3f}".format(np.mean(nll_test[-5:])))
    plt.xlabel('Epoch')
    plt.ylabel(r'NLL')
    plt.legend(loc='lower right')
    plt.savefig(f"{args.results_path}/Test_NLL.pdf", dpi=600)
    plt.close()
    np.savetxt(f"{args.results_path}/Test_NLL_test.txt", nll_test)


def plot_std(samples, epoch):
    plt.imshow(samples, cmap='jet', origin='lower', interpolation='bilinear')
    plt.colorbar()
    plt.tight_layout(pad=0.05, w_pad=0.05, h_pad=0.05)
    plt.savefig(f'{args.results_path}/std_{epoch}.pdf', dpi=300, bbox_inches='tight')
    plt.close()
