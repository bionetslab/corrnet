import numpy as np


def return_fig(fig, save_as):
    fig.tight_layout()
    if save_as:
        fig.savefig(save_as)
    return fig


def normalized(d):
    max_value = np.max(list(d.values()))
    return {key: value / max_value for key, value in d.items()}


def check_attribute(graph, attribute):
    if attribute not in graph.graph['attribute_names']:
        raise ValueError(f'No attribute "{attribute}".')