import networkx as nx
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import json


def temporal_analysis(letter_manager, subjects_as_nodes=True, earliest_date=None, latest_date=None, window_size='5 y',
                      step_width='1 y', save_as=None):
    g = letter_manager.to_digraph(subjects_as_nodes, earliest_date, latest_date)
    window_columns = ['window_start', 'window_end', 'node_preservation', 'edge_preservation']
    graph_columns = ['num_nodes', 'num_edges', 'transitivity', 'mean_size_scc', 'mean_size_wcc', 'coverage_largest_scc',
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
        g = letter_manager.to_digraph(subjects_as_nodes, window_start, window_end)
        new_node_set = set(g.nodes())
        new_edge_set = set(g.edges())
        if len(new_node_set) == 0:
            if window_end >= latest_date:
                break
            window_start, window_end = _increment_window(window_start, window_end, step_width)
            continue
        temporal_data['node_preservation'].append(_jaccard_index(old_node_set, new_node_set))
        temporal_data['edge_preservation'].append(_jaccard_index(old_edge_set, new_edge_set))
        old_node_set = new_node_set
        old_edge_set = new_edge_set
        temporal_data['window_start'].append(window_start)
        temporal_data['window_end'].append(window_end)
        properties = compute_network_properties(g)
        for graph_column in graph_columns:
            temporal_data[graph_column].append(properties[graph_column])
        pageranks = compute_pageranks(g)
        for node in all_nodes:
            temporal_data[f'pagerank_original_{node}'].append(pageranks['original']['pageranks'].get(node, 0.0))
            temporal_data[f'pagerank_reversed_{node}'].append(pageranks['reversed']['pageranks'].get(node, 0.0))
        if window_end >= latest_date:
            break
        window_start, window_end = _increment_window(window_start, window_end, step_width)
    temporal_data = pd.DataFrame(data=temporal_data)
    mean_pageranks = dict()
    mean_pageranks['original'] = sorted([(node, temporal_data[f'pagerank_original_{node}'].mean())
                                         for node in all_nodes], key=lambda t: t[1], reverse=True)
    mean_pageranks['reversed'] = sorted([(node, temporal_data[f'pagerank_reversed_{node}'].mean())
                                         for node in all_nodes], key=lambda t: t[1], reverse=True)
    if save_as:
        temporal_data.to_csv(save_as)
    else:
        return temporal_data, mean_pageranks


def plot_degree_distributions(g, loglog=True, use_weights=False, save_as=None):
    subjects_as_nodes = g.graph['subjects_as_nodes']
    fig, axes = plt.subplots(nrows=3+subjects_as_nodes, ncols=3, figsize=(9+3*subjects_as_nodes, 9))
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
    fig.tight_layout()
    if save_as:
        fig.savefig(save_as)
    else:
        return fig


def compute_pageranks(g, alpha=0.85, k=10, save_as=None):
    pageranks = dict()
    pageranks['original'] = {'pageranks': nx.pagerank_scipy(g, alpha=alpha)}
    pageranks['original'][f'top_{k}'] = sorted(list(pageranks['original']['pageranks'].items()),
                                               key=lambda t: t[1], reverse=True)[:k]
    pageranks['reversed'] = {'pageranks': nx.pagerank_scipy(g.reverse(copy=False), alpha=alpha)}
    pageranks['reversed'][f'top_{k}'] = sorted(list(pageranks['reversed']['pageranks'].items()),
                                               key=lambda t: t[1], reverse=True)[:k]
    if save_as:
        with open(save_as, mode='w') as fp:
            json.dump(pageranks, fp, indent='\t', sort_keys=True)
    else:
        return pageranks


def compute_network_properties(g, save_as=None):
    properties = dict()
    properties['num_nodes'] = nx.number_of_nodes(g)
    properties['num_edges'] = nx.number_of_edges(g)
    properties['transitivity'] = nx.transitivity(g)
    sccs = list(nx.strongly_connected_components(g))
    wccs = list(nx.weakly_connected_components(g))
    properties['mean_size_scc'] = properties['num_nodes'] / sum(1 for _ in sccs)
    properties['mean_size_wcc'] = properties['num_nodes'] / sum(1 for _ in wccs)
    properties['coverage_largest_scc'] = max([len(scc) for scc in sccs]) / properties['num_nodes']
    properties['coverage_largest_wcc'] = max([len(wcc) for wcc in wccs]) / properties['num_nodes']
    _compute_degree_statistics(g, use_weights=False, properties=properties)
    _compute_degree_statistics(g, use_weights=True, properties=properties)
    if save_as:
        with open(save_as, mode='w') as fp:
            json.dump(properties, fp, indent='\t', sort_keys=True)
    else:
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


def _jaccard_index(set_1, set_2):
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

