import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from copy import deepcopy
from easydict import EasyDict as dict


default_config = dict(
    style='seaborn-whitegrid',
    use_latex=False,
    title='Title',
    title_params=dict(size=40, pad=20),
    tick_params=dict(
        x=dict(labelsize=40, pad=10), 
        y=dict(labelsize=40, pad=10)
    ),
    xscale='linear', xscale_params=dict(),
    yscale='linear', yscale_params=dict(),
    xlim=None, ylim=None,
    xlabel='X', xlabel_params=dict(size=40.0), 
    ylabel='Y', ylabel_params=dict(size=40.0),
    size_inches=(16, 12),
    linewidth=10,
    data_cfg=dict(
        data=dict(label=([1,2,3], [1,2,3])),
        linestyles=['-' for _ in range(10)],
    ),
    legend_cfg=dict(
        loc=1, 
        numpoints=1,
        handlelength=2.25,
        prop=dict(size=40.0),
        frameon=True, 
        framealpha=1, 
        facecolor='white', 
        ncol=1
    ),
    save_path='assets/figures/plot.svg'
)

color_set = {
    'Amaranth': np.array([0.9, 0.17, 0.31]),  # main algo
    'Amber': np.array([1.0, 0.49, 0.0]),  # main baseline
    'Bleu de France': np.array([0.19, 0.55, 0.91]),
    'Dark sea green': 'forestgreen',
    'Dark gray': np.array([0.66, 0.66, 0.66]),
    'Arsenic': np.array([0.23, 0.27, 0.29]),
    'Electric violet': np.array([0.56, 0.0, 1.0]),
}

color_list = []
for key, value in color_set.items():
    color_list.append(value)


def plot(cfg: dict) -> None:
    config = deepcopy(default_config)
    config.update(cfg)
    plt.style.use(config.style)
    # plt.rc('font', family='DejaVu Sans')
    # plt.rc('text', usetex=False)
    if config.use_latex:
        plt.rc('text.latex', preamble=r'\usepackage{amsmath}')
    plt.clf()
    plt.tick_params('x', **config.tick_params.x)
    plt.tick_params('y', **config.tick_params.y)

    plt.ylim(config.ylim)
    plt.xlim(config.xlim)
    plt.xlabel(config.xlabel, **config.xlabel_params)
    plt.ylabel(config.ylabel, **config.ylabel_params)

    fig = plt.gcf()
    fig.set_size_inches(*config.size_inches)

    ax = plt.gca()
    ax.set_xscale(config.xscale, **config.xscale_params)
    ax.set_yscale(config.yscale, **config.yscale_params)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_color('black')
    ax.spines['bottom'].set_color('black')

    # ax.xaxis.set_major_locator(ticker.MaxNLocator(6))
    # ax.yaxis.set_major_locator(ticker.MaxNLocator(6))

    plt.title(config.title, **config.title_params)

    for (label, data), linestyle, color in zip(
        config.data_cfg.data.items(), 
        config.data_cfg.linestyles,
        color_list, 
    ):
        # print(data)
        ax.plot(
            *data,
            label=label, 
            color=color, 
            linestyle=linestyle, 
            linewidth=config.linewidth
        )

    plt.legend(**config.legend_cfg)
    plt.savefig(config.save_path)
