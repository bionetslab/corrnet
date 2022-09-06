import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from statannotations.Annotator import Annotator
import itertools as itt
import corrnet.utils as utils
from corrnet.compute_centralities import compute_centralities


def differential_centrality_analysis(digraph, multi_digraph, split_attribute, direction,
                                     centrality_measures = ['Degree centrality', 'Harmonic centrality', 'PageRank centrality'],
                                     roles_as_columns=False, annotate=True, test='Mann-Whitney', text_format='star',
                                     figsize=None, save_as=None):

    # Ensure that graph has split attribute.
    utils.check_attribute(multi_digraph, split_attribute)

    # Prepare the figure and the node sets.
    fig, axes = _setup_digraph_figure(centrality_measures, roles_as_columns, figsize)
    sender_sets, addressee_sets, found_values = _split_digraph_senders_and_addressees(multi_digraph, split_attribute)

    # Carry out the analyses and plot the results.
    for centrality_id, centrality_measure in enumerate(centrality_measures):
        for node_sets, role_id, role in zip([sender_sets, addressee_sets], [0, 1],
                                            [multi_digraph.graph['sender_attribute_name'],
                                             multi_digraph.graph['addressee_attribute_name']]):
            df, centrality_col_name = _compute_centralities(digraph, centrality_measure, direction, found_values,
                                                            node_sets, split_attribute)
            axis = _get_axis(centrality_measures, roles_as_columns, axes, centrality_id, role_id)
            _plot_axis(df, centrality_col_name, split_attribute, axis, f'{role} nodes', annotate, found_values, test,
                       text_format)

    # Return the figure.
    utils.return_fig(fig, save_as)


def differential_line_graph_centrality_analysis(line_graph, split_attribute, direction,
                                                centrality_measures=['Degree centrality', 'Harmonic centrality',
                                                                     'PageRank centrality'], annotate=True,
                                                test='Mann-Whitney', text_format='star', figsize=None, save_as=None):

    # Ensure that graph has split attribute.
    utils.check_attribute(line_graph, split_attribute)

    # Prepare the figure and the node sets.
    fig, axes = _setup_line_graph_figure(centrality_measures, figsize)
    node_sets, found_values = _split_line_graph_senders_and_addressees(line_graph, split_attribute)

    # Carry out the analyses and plot the results.
    for centrality_id, centrality_measure in enumerate(centrality_measures):
        df, centrality_col_name = _compute_centralities(line_graph, centrality_measure, direction, found_values,
                                                        node_sets, split_attribute)
        _plot_axis(df, centrality_col_name, split_attribute, axes[centrality_id], None, annotate,
                   found_values, test, text_format)

    # Return the figure.
    utils.return_fig(fig, save_as)


def _setup_digraph_figure(centrality_measures, roles_as_columns, figsize):
    if figsize is None:
        if roles_as_columns:
            figsize = (6, 3 * len(centrality_measures))
        else:
            figsize = (3 * len(centrality_measures), 6)
    if roles_as_columns:
        return plt.subplots(nrows=len(centrality_measures), ncols=2, figsize=figsize)
    else:
        return plt.subplots(nrows=2, ncols=len(centrality_measures), figsize=figsize)


def _setup_line_graph_figure(centrality_measures, figsize):
    if figsize is None:
        figsize = (3 * len(centrality_measures), 3)
    return plt.subplots(nrows=1, ncols=len(centrality_measures), figsize=figsize)


def _split_digraph_senders_and_addressees(multi_digraph, split_attribute):
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
    return sender_sets, addressee_sets, found_values


def _split_line_graph_senders_and_addressees(line_graph, split_attribute):
    node_sets = dict()
    found_values = set()
    for node in line_graph.nodes(data=True):
        value = node[1][split_attribute]
        if pd.isna(value):
            continue
        value = str(value)
        if value not in found_values:
            node_sets[value] = set()
            found_values.add(value)
        node_sets[value].add(node[0])
    return node_sets, found_values


def _compute_centralities(digraph, centrality_measure, direction, found_values, node_sets, split_attribute):
    centralities = compute_centralities(digraph, centrality_measure, direction)
    all_attributes = []
    all_centralities = []
    for value in found_values:
        for node in node_sets[value]:
            all_attributes.append(value)
            all_centralities.append(centralities[node])
    centrality_col_name = f'{centrality_measure} ({direction})'
    df = pd.DataFrame(data={centrality_col_name: all_centralities, split_attribute: all_attributes})
    return df, centrality_col_name


def _get_axis(centrality_measures, roles_as_columns, axes, centrality_id, role_id):
    if len(centrality_measures) > 1:
        if roles_as_columns:
            return axes[centrality_id, role_id]
        else:
            return axes[role_id, centrality_id]
    return axes[role_id]


def _plot_axis(df, centrality_col_name, split_attribute, axis, title, annotate, found_values, test, text_format):
    sns.violinplot(data=df, y=centrality_col_name, x=split_attribute, cut=0, ax=axis)
    if title is not None:
        axis.set_title(title)
    if annotate:
        pairs = list(itt.combinations(found_values, 2))
        annotator = Annotator(axis, pairs, data=df, y=centrality_col_name, x=split_attribute, plot='violinplot',
                              verbose=True)
        annotator.configure(test=test, text_format=text_format, loc='inside', pvalue_format_string='{:.2e}')
        annotator.apply_and_annotate()
