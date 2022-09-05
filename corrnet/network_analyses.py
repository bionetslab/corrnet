import networkx as nx
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import json
from statannotations.Annotator import Annotator
import itertools as itt


def temporal_analysis(letter_manager, earliest_date=None, latest_date=None, filter_by=None,
                      window_size='5 y', step_width='1 y', save_as=None):
    digraph, _ = letter_manager.to_graph(earliest_date, latest_date, build_multi_digraph=False, filter_by=filter_by)
    window_columns = ['window_start', 'window_end', 'node_preservation', 'edge_preservation', 'node_novelty',
                      'edge_novelty', 'node_congruence', 'edge_congruence']
    graph_columns = ['num_nodes', 'num_edges', 'transitivity', 'num_sccs', 'num_wccs', 'coverage_largest_scc',
                     'coverage_largest_wcc']
    graph_columns += [f'{prefix}_{infix}_degree_{suffix}' for prefix in ['weighted', 'unweighted']
                      for infix in ['in', 'out', 'total'] for suffix in ['mean', 'median', 'var', 'max', 'min']]
    all_nodes = list(digraph.nodes())
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
        digraph = letter_manager.to_graph(window_start, window_end, build_multi_digraph=False, filter_by=filter_by)
        new_node_set = set(digraph.nodes())
        new_edge_set = set(digraph.edges())
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
        properties = compute_network_properties(digraph)
        for graph_column in graph_columns:
            temporal_data[graph_column].append(properties[graph_column])
        pageranks = compute_centralities(digraph)
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
    if figsize is None:
        figsize = (9, 9)
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=figsize)
    nodes = list(g.nodes())
    _plot_degree_distributions_for_nodes(g, nodes, 'All nodes', 0, loglog, use_weights, axes)
    sender_col = g.graph['sender_col']
    nodes = [node for node in g.nodes() if sender_col in g.nodes[node]['roles']]
    _plot_degree_distributions_for_nodes(g, nodes, f'"{sender_col}" nodes', 1, loglog, use_weights, axes)
    addressee_col = g.graph['addressee_col']
    nodes = [node for node in g.nodes() if addressee_col in g.nodes[node]['roles']]
    _plot_degree_distributions_for_nodes(g, nodes, f'"{addressee_col}" nodes', 2, loglog, use_weights, axes)
    _return_fig(fig, save_as)


def compute_centralities(digraph, centrality_measure='pagerank', direction='in', normalize=False, save_as=None):
    centrality_fun = _get_centrality_fun(centrality_measure)
    centralities = None
    if direction == 'in':
        centralities = _compute_centralities(digraph, centrality_fun, normalize)
    if direction == 'out':
        centralities = _compute_centralities(digraph.reverse(copy=False), centrality_fun, normalize)
    if save_as:
        with open(save_as, mode='w') as fp:
            json.dump(centralities, fp, indent='\t', sort_keys=True)
    return centralities


def differential_centrality_analysis(digraph, multi_digraph, split_attribute, centrality_measures, direction,
                                     roles_as_columns=True, annotate=True, test='Mann-Whitney', text_format='full',
                                     figsize=None, save_as=None):
    if figsize is None:
        if roles_as_columns:
            figsize = (6, 3 * len(centrality_measures))
        else:
            figsize = (3 * len(centrality_measures), 6)
    if roles_as_columns:
        fig, axes = plt.subplots(nrows=len(centrality_measures), ncols=2, figsize=figsize)
    else:
        fig, axes = plt.subplots(nrows=2, ncols=len(centrality_measures), figsize=figsize)

    sender_sets = dict()
    addressee_sets = dict()
    found_values = set()
    for edge in multi_digraph.edges(data=True):
        value = edge[2][split_attribute]
        if pd.isna(value):
            continue
        value = str(value)
        if value not in found_values:
            sender_sets[value] = set()
            addressee_sets[value] = set()
            found_values.add(value)
        sender_sets[value].add(edge[0])
        addressee_sets[value].add(edge[1])

    for centrality_id, centrality_measure in enumerate(centrality_measures):
        for node_sets, role_id, role in zip([sender_sets, addressee_sets], [0, 1],
                                           [multi_digraph.graph['sender_col'], multi_digraph.graph['addressee_col']]):
            centralities = compute_centralities(digraph, centrality_measure, direction)
            all_attributes = []
            all_centralities = []
            for value in found_values:
                for node in node_sets[value]:
                    all_attributes.append(value)
                    all_centralities.append(centralities[node])
            centrality_col_name = f'{centrality_measure} ({direction})'
            df = pd.DataFrame(data={centrality_col_name: all_centralities, split_attribute: all_attributes})
            if len(centrality_measures) > 1:
                if roles_as_columns:
                    axis = axes[centrality_id, role_id]
                else:
                    axis = axes[role_id, centrality_id]
            else:
                axis = axes[role_id]
            sns.violinplot(data=df, y=centrality_col_name, x=split_attribute, cut=0, ax=axis)
            axis.set_title(f'{role} nodes')
            if annotate:
                pairs = list(itt.combinations(found_values, 2))
                annotator = Annotator(axis, pairs, data=df, y=centrality_col_name, x=split_attribute, plot='violinplot',
                                      verbose=True)
                annotator.configure(test=test, text_format=text_format, loc='inside', pvalue_format_string='{:.2e}')
                annotator.apply_and_annotate()

    _return_fig(fig, save_as)




def _get_centrality_fun(centrality_measure):
    centrality_fun_dict = {
        'PageRank centrality': nx.pagerank_scipy,
        'Harmonic centrality': nx.centrality.harmonic_centrality,
        'Betweenness centrality': nx.centrality.betweenness_centrality,
        'Degree centrality': nx.centrality.in_degree_centrality
    }
    return centrality_fun_dict[centrality_measure]


def _compute_centralities(digraph, centrality_fun, normalize):
    centralities = centrality_fun(digraph)
    if normalize:
        centralities = _normalized(centralities)
    return centralities


def compute_network_properties(digraph, save_as=None):
    properties = dict()
    properties['num_nodes'] = nx.number_of_nodes(digraph)
    properties['num_edges'] = nx.number_of_edges(digraph)
    properties['transitivity'] = nx.transitivity(digraph)
    sccs = list(nx.strongly_connected_components(digraph))
    wccs = list(nx.weakly_connected_components(digraph))
    properties['num_sccs'] = sum(1 for _ in sccs)
    properties['num_wccs'] = sum(1 for _ in wccs)
    properties['coverage_largest_scc'] = max([len(scc) for scc in sccs]) / properties['num_nodes']
    properties['coverage_largest_wcc'] = max([len(wcc) for wcc in wccs]) / properties['num_nodes']
    _compute_degree_statistics(digraph, use_weights=False, properties=properties)
    _compute_degree_statistics(digraph, use_weights=True, properties=properties)
    if save_as:
        with open(save_as, mode='w') as fp:
            json.dump(properties, fp, indent='\t', sort_keys=True)
    return properties


def _plot_degree_distributions_for_nodes(digraph, nodes, title, row, loglog, use_weights, axes):
    weight = None
    if use_weights:
        weight = 'weight'
    degrees = [_total_degree(digraph, weight)[node] for node in nodes]
    _plot_degree_distribution(degrees, title, 'Total degree', loglog, axes[row, 0])
    degrees = [digraph.in_degree(node, weight) for node in nodes]
    _plot_degree_distribution(degrees, title, 'In-degree', loglog, axes[row, 1])
    degrees = [digraph.out_degree(node, weight) for node in nodes]
    _plot_degree_distribution(degrees, title, 'Out-degree', loglog, axes[row, 2])


def _total_degree(digraph, weight):
    in_degrees = digraph.in_degree(weight)
    out_degrees = digraph.out_degree(weight)
    return {node: in_degrees[node] + out_degrees[node] for node in digraph.nodes()}


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
