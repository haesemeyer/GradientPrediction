#  Copyright 2017 Martin Haesemeyer. All rights reserved.
#
# Licensed under the MIT license

"""
Builds a gradient prediction network that uses a mixed input layer, i.e. the input is a 2D tensor of both behavior and
temperature variables on which convolution filters operate jointly
"""

import core
import numpy as np
from global_defs import GlobalDefs


if __name__ == "__main__":
    import matplotlib.pyplot as pl
    import seaborn as sns
    from scipy.ndimage import gaussian_filter1d
    N_CONV_LAYERS = 40
    print("Testing mixedInputModel", flush=True)
    print("For each 'behavior' subpart attempt to learn different sums on standard normal distribution", flush=True)
    t_losses = []
    d_fracs = []
    with core.ZfGpNetworkModel() as model:
        model.setup(N_CONV_LAYERS, 512, 0, 3)
        for i in range(2000):
            xb1 = np.random.randn(100, 1, GlobalDefs.frame_rate * GlobalDefs.hist_seconds, 1)
            xb2 = xb1 ** 2
            xb2 -= 1  # expected average of xb1**2
            xb3 = xb1 ** 3
            xbatch = np.concatenate((xb1, xb2, xb3), 1)
            ybatch = np.c_[np.sum(xb2, axis=(1, 2)), np.sum(xb2 / 4, axis=(1, 2)),
                           np.sum(xb1, axis=(1, 2)), np.sum(xb1 / 2, axis=(1, 2))]
            cur_l = model.get_squared_loss(xbatch, ybatch)
            pred = model.predict(xbatch)
            cur_d = np.median(np.abs((ybatch - pred) / ybatch))
            t_losses.append(cur_l)
            d_fracs.append(cur_d)
            if i % 200 == 0:
                print('step %d, training loss %g, delta fraction %g' % (i, cur_l, cur_d))
            model.train(xbatch, ybatch)
        weights_conv, bias_conv = model.convolution_data

    weights_conv = weights_conv['m']
    w_ext = np.max(np.abs(weights_conv))
    fig, ax = pl.subplots(ncols=int(np.sqrt(N_CONV_LAYERS)), nrows=int(np.sqrt(N_CONV_LAYERS)), frameon=False,
                          figsize=(14, 2.8))
    ax = ax.ravel()
    for i, a in enumerate(ax):
        sns.heatmap(weights_conv[:, :, 0, i], ax=a, vmin=-w_ext, vmax=w_ext, center=0, cbar=False)
        a.axis("off")

    pl.figure()
    pl.plot(t_losses, 'o')
    pl.xlabel("Batch")
    pl.ylabel("Training loss")
    sns.despine()

    pl.figure()
    pl.plot(d_fracs, 'o')
    pl.plot(gaussian_filter1d(d_fracs, 25))
    pl.xlabel("Batch")
    pl.ylabel("Error fraction")
    sns.despine()
