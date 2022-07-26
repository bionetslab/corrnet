import networkx as nx
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import json


def temporal_analysis(letter_manager, subjects_as_nodes=False, earliest_date=None, latest_date=None, type_filters=None,
                      window_size='5 y', step_width='1 y', save_as=None):
    g = letter_manager.to_digraph(subjects_as_nodes, earliest_date, latest_date, type_filters)
    window_columns = ['window_start', 'window_end', 'node_preservation', 'edge_preservation', 'node_novelty',
                      'edge_novelty', 'node_congruence', 'edge_congruence']
    graph_columns = ['num_nodes', 'num_edges', 'transitivity', 'num_sccs', 'num_wccs', 'coverage_largest_scc',
                     'coverage_largest_wcc']
    graph_columns += [f'{prefix}_{infix}_degree_{suffix}' for prefix in ['weighted', 'unweighted']
                      for infix in ['in', 'out', 'total'] for suffix in ['mean', 'median', 'var', 'max', 'min']]
    all_nodes = list(g.nodes())
    node_columns = [f'pagerank_original_{node}' for node in all_nodes]
    node_columns += [f'pagerank_reversed_{node}' for node in all_nodes]
    columns = window_columns + graph_columns + node_columns
    temporal_data = {column: [] for column in columns}
    earliest_date, latest_date = letter_manager.init_earliest_and_latest_date(earliest_date, latest_date)
    window_start = earliest_date
    window_size = _get_num_months(window_size, 'window_size')
    step_width = _get_num_months(step_width, 'step_width')
    window_end = _add_months_to_timestamp(earliest_date, window_size)
    old_node_set = None
    old_edge_set = None
    while True:
        g = letter_manager.to_digraph(subjects_as_nodes, window_start, window_end, type_filters)
        new_node_set = set(g.nodes())
        new_edge_set = set(g.edges())
        if len(new_node_set) == 0:
            if window_end >= latest_date:
                break
            window_start, window_end = _increment_window(window_start, window_end, step_width)
            continue
        temporal_data['node_preservation'].append(_preservation(old_node_set, new_node_set))
        temporal_data['edge_preservation'].append(_preservation(old_edge_set, new_edge_set))
        temporal_data['node_novelty'].append(_novelty(old_node_set, new_node_set))
        temporal_data['edge_novelty'].append(_novelty(old_edge_set, new_edge_set))
        temporal_data['node_congruence'].append(_congruence(old_node_set, new_node_set))
        temporal_data['edge_congruence'].append(_congruence(old_edge_set, new_edge_set))
        old_node_set = new_node_set
        old_edge_set = new_edge_set
        temporal_data['window_start'].append(window_start)
        temporal_data['window_end'].append(window_end)
        properties = compute_network_properties(g)
        for graph_column in graph_columns:
            temporal_data[graph_column].append(properties[graph_column])
        pageranks = compute_pagerank(g)
        for node in all_nodes:
            temporal_data[f'pagerank_original_{node}'].append(pageranks['original']['pageranks'].get(node, 0.0))
            temporal_data[f'pagerank_reversed_{node}'].append(pageranks['reversed']['pageranks'].get(node, 0.0))
        if window_end >= latest_date:
            break
        window_start, window_end = _increment_window(window_start, window_end, step_width)
    temporal_data = pd.DataFrame(data=temporal_data)
    if save_as:
        temporal_data.to_csv(save_as)
    return temporal_data


def plot_pagerank(temporal_data, figsize=None, nodes=None, save_as=None):
    if nodes is None:
        aggregated_pageranks = sort_nodes_by_aggregated_pagerank(temporal_data)
        nodes = [aggregated_pageranks['original'][0][0], aggregated_pageranks['reversed'][0][0]]
    if figsize is None:
        figsize = (3*len(nodes), 6)
    fig, axes = plt.subplots(nrows=len(nodes), ncols=2, figsize=figsize)
    for i, node in enumerate(nodes):
        sns.lineplot(data=temporal_data, x='window_start', y=f'pagerank_original_{node}', ax=axes[i,0])
        sns.lineplot(data=temporal_data, x='window_start', y=f'pagerank_reversed_{node}', ax=axes[i, 1])
        for j in [0,1]:
            axes[i, j].set_xlabel('Window start')
            axes[i, j].set_title(node)
        axes[i, 0].set_ylabel('Normalized PageRank\non original network')
        axes[i, 1].set_ylabel('Normalized PageRank\non reversed network')
    _return_fig(fig, save_as)


def plot_network_properties(temporal_data, figsize=None, save_as=None):
    if figsize is None:
        figsize = (9, 9)
    graph_columns = ['num_nodes', 'num_edges', 'transitivity', 'num_sccs', 'num_wccs', 'coverage_largest_scc',
                     'coverage_largest_wcc']
    ylabels = {'num_nodes': 'Number of\nnodes', 'num_edges': 'Number of\nedges', 'transitivity': 'Transitivity',
               'num_sccs': 'Number of\nSCCs', 'num_wccs': 'Number of\nWCCs',
               'coverage_largest_scc': 'Coverage of\nlargest SCC', 'coverage_largest_wcc': 'Coverage of\nlargest WCC'}
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=figsize)
    for k, column in enumerate(graph_columns):
        i = k // 3
        j = k % 3
        sns.lineplot(data=temporal_data, x='window_start', y=column, ax=axes[i, j])
        axes[i, j].set_xlabel('Window start')
        axes[i, j].set_ylabel(ylabels[column])
    axes[2, 1].axis('off')
    axes[2, 2].axis('off')
    _return_fig(fig, save_as)


def plot_network_dynamics(temporal_data, figsize=None, save_as=None):
    if figsize is None:
        figsize = (9, 6)
    window_columns = ['node_preservation', 'node_novelty', 'node_congruence', 'edge_preservation', 'edge_novelty',
                      'edge_congruence']
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=figsize)
    for k, column in enumerate(window_columns):
        i = k // 3
        j = k % 3
        sns.lineplot(data=temporal_data, x='window_start', y=column, ax=axes[i, j])
        axes[i, j].set_xlabel('Window start')
        axes[i, j].set_ylabel(' '.join(column.split('_')).capitalize())
    _return_fig(fig, save_as)


def plot_neighborhood(g, node, figsize=None, save_as=None, margins=(None, None)):
    if figsize is None:
        figsize = (12, 6)
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=figsize)
    in_nbs = list(g.predecessors(node))
    _plot_neighbors(g, node, in_nbs, axes[0], margins[0])
    out_nbs = list(g.successors(node))
    _plot_neighbors(g, node, out_nbs, axes[1], margins[1])
    axes[0].set_title('In-neighbors')
    axes[1].set_title('Out-neighbors')
    _return_fig(fig, save_as)


def _plot_neighbors(g, node, neighbors, ax, margins):
    h = nx.induced_subgraph(g, [node] + neighbors)
    colors = ['red' if u == node else 'cyan' for u in list(h)]
    shells = [[node], neighbors]
    pos = nx.shell_layout(h, shells)
    nx.draw_networkx(h, pos=pos, ax=ax, node_color=colors, font_size=9, margins=margins, edge_color='grey')


def sort_nodes_by_aggregated_pagerank(temporal_data, aggregator=np.mean):
    all_nodes = []
    for column in temporal_data.columns:
        if column.startswith('pagerank_original_'):
            all_nodes.append(column[18:])
    aggregated_pageranks = dict()
    aggregated_pageranks['original'] = sorted([(node, aggregator(temporal_data[f'pagerank_original_{node}']))
                                               for node in all_nodes], key=lambda t: t[1], reverse=True)
    aggregated_pageranks['reversed'] = sorted([(node, aggregator(temporal_data[f'pagerank_reversed_{node}']))
                                               for node in all_nodes], key=lambda t: t[1], reverse=True)
    return aggregated_pageranks


def plot_degree_distributions(g, loglog=True, use_weights=False, figsize=None, save_as=None):
    subjects_as_nodes = g.graph['subjects_as_nodes']
    if figsize is None:
        figsize = (9+3*subjects_as_nodes, 9)
    fig, axes = plt.subplots(nrows=3+subjects_as_nodes, ncols=3, figsize=figsize)
    nodes = list(g.nodes())
    _plot_degree_distributions_for_nodes(g, nodes, 'All nodes', 0, loglog, use_weights, axes)
    sender_col = g.graph['sender_col']
    nodes = [node for node in g.nodes() if sender_col in g.nodes[node]['roles']]
    _plot_degree_distributions_for_nodes(g, nodes, f'"{sender_col}" nodes', 1, loglog, use_weights, axes)
    addressee_col = g.graph['addressee_col']
    nodes = [node for node in g.nodes() if addressee_col in g.nodes[node]['roles']]
    _plot_degree_distributions_for_nodes(g, nodes, f'"{addressee_col}" nodes', 2, loglog, use_weights, axes)
    if subjects_as_nodes:
        subject_col = g.graph['subject_col']
        nodes = [node for node in g.nodes() if subject_col in g.nodes[node]['roles']]
        _plot_degree_distributions_for_nodes(g, nodes, f'"{subject_col}" nodes', 3, loglog, use_weights, axes)
    _return_fig(fig, save_as)


def compute_pagerank(g, alpha=0.85, k=10, save_as=None):
    pageranks = dict()
    pageranks['original'] = {'pageranks': _normalized(nx.pagerank_scipy(g, alpha=alpha))}
    pageranks['original'][f'top_{k}'] = sorted(list(pageranks['original']['pageranks'].items()),
                                               key=lambda t: t[1], reverse=True)[:k]
    pageranks['reversed'] = {'pageranks': _normalized(nx.pagerank_scipy(g.reverse(copy=False), alpha=alpha))}
    pageranks['reversed'][f'top_{k}'] = sorted(list(pageranks['reversed']['pageranks'].items()),
                                               key=lambda t: t[1], reverse=True)[:k]
    if save_as:
        with open(save_as, mode='w') as fp:
            json.dump(pageranks, fp, indent='\t', sort_keys=True)
    return pageranks


def compute_network_properties(g, save_as=None):
    properties = dict()
    properties['num_nodes'] = nx.number_of_nodes(g)
    properties['num_edges'] = nx.number_of_edges(g)
    properties['transitivity'] = nx.transitivity(g)
    sccs = list(nx.strongly_connected_components(g))
    wccs = list(nx.weakly_connected_components(g))
    properties['num_sccs'] = sum(1 for _ in sccs)
    properties['num_wccs'] = sum(1 for _ in wccs)
    properties['coverage_largest_scc'] = max([len(scc) for scc in sccs]) / properties['num_nodes']
    properties['coverage_largest_wcc'] = max([len(wcc) for wcc in wccs]) / properties['num_nodes']
    _compute_degree_statistics(g, use_weights=False, properties=properties)
    _compute_degree_statistics(g, use_weights=True, properties=properties)
    if save_as:
        with open(save_as, mode='w') as fp:
            json.dump(properties, fp, indent='\t', sort_keys=True)
    return properties


def _plot_degree_distributions_for_nodes(g, nodes, title, row, loglog, use_weights, axes):
    weight = None
    if use_weights:
        weight = 'weight'
    degrees = [_total_degree(g, weight)[node] for node in nodes]
    _plot_degree_distribution(degrees, title, 'Total degree', loglog, axes[row, 0])
    degrees = [g.in_degree(node, weight) for node in nodes]
    _plot_degree_distribution(degrees, title, 'In-degree', loglog, axes[row, 1])
    degrees = [g.out_degree(node, weight) for node in nodes]
    _plot_degree_distribution(degrees, title, 'Out-degree', loglog, axes[row, 2])


def _total_degree(g, weight):
    in_degrees = g.in_degree(weight)
    out_degrees = g.out_degree(weight)
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
    if month == 2 and day >= 29:
        day = 28
    return pd.to_datetime(f'{year}-{month}-{day}')


def _add_months_to_timestamp(timestamp, num_months):
    num_months = timestamp.month + num_months
    num_years = num_months // 12
    timestamp = _add_years_to_timestamp(timestamp, num_years)
    num_months = num_months % 12
    year = timestamp.year
    month = num_months
    day = timestamp.day
    if month == 2 and day >= 29:
        day = 28
    if month in [2, 4, 6, 9, 11] and day >= 31:
        day = 30
    return pd.to_datetime(f'{year}-{month}-{day}')


def _compute_degree_statistics(g, use_weights, properties):
    weight = None
    prefix = 'unweighted'
    if use_weights:
        weight = 'weight'
        prefix = 'weighted'
    in_degrees = g.in_degree(weight)
    out_degrees = g.out_degree(weight)
    total_degrees = np.array([in_degrees[node] + out_degrees[node] for node in g.nodes()])
    in_degrees = np.array([in_degrees[node] for node in g.nodes()])
    out_degrees = np.array([out_degrees[node] for node in g.nodes()])
    _summarize_degree_sequence(total_degrees, f'{prefix}_total', properties)
    _summarize_degree_sequence(in_degrees, f'{prefix}_in', properties)
    _summarize_degree_sequence(out_degrees, f'{prefix}_out', properties)


def _summarize_degree_sequence(degrees, prefix, properties):
    properties[f'{prefix}_degree_mean'] = float(np.mean(degrees))
    properties[f'{prefix}_degree_median'] = float(np.median(degrees))
    properties[f'{prefix}_degree_var'] = float(np.var(degrees))
    properties[f'{prefix}_degree_max'] = int(np.max(degrees))
    properties[f'{prefix}_degree_min'] = int(np.min(degrees))


def _preservation(set_1, set_2):
    if set_1 is None or set_2 is None:
        return None
    return len(set_1.intersection(set_2)) / len(set_1)


def _novelty(set_1, set_2):
    if set_1 is None or set_2 is None:
        return None
    return len(set_2.difference(set_1)) / len(set_2)


def _congruence(set_1, set_2):
    if set_1 is None or set_2 is None:
        return None
    return len(set_1.intersection(set_2)) / len(set_1.union(set_2))


def _get_num_months(interval_size, parameter_name):
    interval_size_info = interval_size.split(' ')
    if interval_size_info[1] == 'y':
        return int(interval_size_info[0]) * 12
    elif interval_size_info[1] == 'm':
        return int(interval_size_info[0])
    else:
        msg = f'Invalid {parameter_name} {interval_size}. Required format: "<number of years or months> [y|m]'
        raise ValueError(msg)


def _increment_window(window_start, window_end, step_width):
    window_start = _add_months_to_timestamp(window_start, step_width)
    window_end = _add_months_to_timestamp(window_end, step_width)
    return window_start, window_end


def _return_fig(fig, save_as):
    fig.tight_layout()
    if save_as:
        fig.savefig(save_as)
    return fig


def _normalized(d):
    max_value = np.max(list(d.values()))
    return {key: value / max_value for key, value in d.items()}
