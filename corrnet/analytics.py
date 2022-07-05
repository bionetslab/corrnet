import networkx as nx
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter


def plot_degree_distributions(g, loglog=True, save_as=None):
    subjects_as_nodes = g.graph['subjects_as_nodes']
    fig, axes = plt.subplots(nrows=3+subjects_as_nodes, ncols=3, figsize=(9+3*subjects_as_nodes, 9))
    nodes = list(g.nodes())
    _plot_degree_distributions_for_nodes(g, nodes, 'All nodes', 0, loglog, axes)
    sender_col = g.graph['sender_col']
    nodes = [node for node in g.nodes() if sender_col in g.nodes[node]['roles']]
    _plot_degree_distributions_for_nodes(g, nodes, f'"{sender_col}" nodes', 1, loglog, axes)
    addressee_col = g.graph['addressee_col']
    nodes = [node for node in g.nodes() if addressee_col in g.nodes[node]['roles']]
    _plot_degree_distributions_for_nodes(g, nodes, f'"{addressee_col}" nodes', 2, loglog, axes)
    if subjects_as_nodes:
        subject_col = g.graph['subject_col']
        nodes = [node for node in g.nodes() if subject_col in g.nodes[node]['roles']]
        _plot_degree_distributions_for_nodes(g, nodes, f'"{subject_col}" nodes', 3, loglog, axes)
    fig.tight_layout()
    if save_as:
        fig.savefig(save_as)
    else:
        return fig


def _plot_degree_distributions_for_nodes(g, nodes, title, row, loglog, axes):
    degrees = [_total_degree(g)[node] for node in nodes]
    _plot_degree_distribution(degrees, title, 'Total degree', loglog, axes[row, 0])
    degrees = [g.in_degree()[node] for node in nodes]
    _plot_degree_distribution(degrees, title, 'In-degree', loglog, axes[row, 1])
    degrees = [g.out_degree()[node] for node in nodes]
    _plot_degree_distribution(degrees, title, 'Out-degree', loglog, axes[row, 2])


def _total_degree(g):
    in_degrees = g.in_degree()
    out_degrees = g.out_degree()
    return {node: in_degrees[node] + out_degrees[node] for node in g.nodes()}


def _plot_degree_distribution(degrees, title, xlabel, loglog, ax):
    degree_counts = Counter(degrees)
    x, y = zip(*degree_counts.items())
    ax.scatter(x, y, marker='.')
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Frequency')
    if loglog:
        ax.set_xscale('log')
        ax.set_yscale('log')

def _add_years_to_timestamp(timestamp, num_years):
    year = timestamp.year + num_years
    month = timestamp.month
    day = timestamp.day
    if month == 2 and day == 29:
        day = 28
    return pd.to_datetime(f'{year}-{month}-{day}')