import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from statannotations.Annotator import Annotator
import itertools as itt
import corrnet.utils as utils
from corrnet.compute_centralities import compute_centralities


def differential_centrality_analysis(digraph, multi_digraph, split_attribute, direction,
                                     centrality_measures = ['Degree centrality', 'Closeness centrality',
                                                            'Harmonic centrality', 'PageRank centrality'], roles_as_columns=False,
                                     annotate=True, figsize=None, save_as=None, direction_in_ylabel=True,
                                     plot='violinplot'):
    """Function to carry out differential node centrality analyses on a digraph representation.

    Carries out differential node centrality analyses for sender- and receiver-nodes incident with edges with
    differing ``split_attribute`` values.

    Args:
        digraph (networkx.DiGraph): Directed correspondence network (edges from senders to receivers, edges weighted by
            multiplicity).
        multi_digraph (networkx.MultiDiGraph): Multigraph representation of directed correspondence network (edges from
            senders to receivers).
        split_attribute (str): The edge attribute of ``multi_graph`` used to split the edges.
        direction (str): Specifies direction of centralities; 'in' for in-centralities, 'out' for out-centralities.
        centrality_measures (list of str): Specifies centrality measure. List may contain 'Degree centrality',
            'Harmonic centrality', 'PageRank centrality', 'Closeness centrality', and 'Betweenness centrality'.
        roles_as_columns (bool): True if node roles should be displayed in the columns of the returned violin plots,
            False if they should be displayed in the rows.
        annotate (bool): If True, the violin plots are decorated with statistical significance annotations.
        figsize (tuple or None): Size of the returned figure. If None, the size is determined automatically.
        save_as (str or None): If provided, the returned figure is saved at this path.
        direction_in_ylabel (True): If True, the direction of the centralities is displayed in ylabel of generated plot.
        plot (str): Type of plot to be generated, either 'violinplot' or 'boxplot'.

    Returns:
        matplotlib.figure.Figure: A figure visualizing the results of the analysis.
        pandas.DataFrame: A dataframe with the results of the analysis.
    """

    # Ensure that graph has split attribute.
    utils.check_attribute(multi_digraph, split_attribute)

    # Prepare the figure and the node sets.
    fig, axes = _setup_digraph_figure(centrality_measures, roles_as_columns, figsize)
    sender_sets, addressee_sets, found_values = _split_digraph_senders_and_addressees(multi_digraph, split_attribute)

    # Carry out the analyses and plot the results.
    for centrality_id, centrality_measure in enumerate(centrality_measures):
        if direction_in_ylabel:
            centrality_col_name = f'{centrality_measure} ({direction})'
        else:
            centrality_col_name = centrality_measure
        for node_sets, role_id, role in zip([sender_sets, addressee_sets], [0, 1],
                                            [multi_digraph.graph['sender_attribute_name'],
                                             multi_digraph.graph['addressee_attribute_name']]):
            df = _compute_centralities(digraph, centrality_measure, direction,
                                                                           found_values, node_sets, split_attribute,
                                                                           centrality_col_name)
            axis = _get_axis(centrality_measures, roles_as_columns, axes, centrality_id, role_id)
            _plot_axis(df, centrality_col_name, split_attribute, axis, f'{role} nodes', annotate, found_values, plot)

    # Return the figure.
    return utils.return_fig(fig, save_as), df


def differential_line_graph_centrality_analysis(line_graph, split_attribute, direction,
                                                centrality_measures=['Degree centrality', 'Closeness centrality',
                                                                     'Harmonic centrality', 'PageRank centrality'], annotate=True,
                                                figsize=None, save_as=None, direction_in_ylabel=True,
                                                plot='violinplot'):
    """Function to carry out differential node centrality analyses on a line graph representation.

    Carries out differential node centrality analyses for line graph nodes (i.e., edges in digraph representation) with
    differing ``split_attribute`` values.

    Args:
        line_graph (nx.DiGraph): Line graph representation of directed correspondence network.
        split_attribute (str): The node attribute of `line_graph` used to split the nodes.
        direction (str): Specifies direction of centralities; 'in' for in-centralities, 'out' for out-centralities.
        centrality_measures (list of str): Specifies centrality measure. List may contain 'Degree centrality',
            'Harmonic centrality', 'PageRank centrality', and 'Betweenness centrality'.
        annotate (bool): If True, the violin plots are decorated with statistical significance annotations.
        figsize (tuple or None): Size of the returned figure. If None, the size is determined automatically.
        save_as (str or None): If provided, the returned figure is saved at this path.
        direction_in_ylabel (True): If True, the direction of the centralities is displayed in ylabel of generated plot.
        plot (str): Type of plot to be generated, either 'violinplot' or 'boxplot'.

    Returns:
        matplotlib.figure.Figure: A figure visualizing the results of the analysis.
        pandas.DataFrame: A dataframe with the results of the analysis.
    """

    # Ensure that graph has split attribute.
    utils.check_attribute(line_graph, split_attribute)

    # Prepare the figure and the node sets.
    fig, axes = _setup_line_graph_figure(centrality_measures, figsize)
    node_sets, found_values = _split_line_graph_senders_and_addressees(line_graph, split_attribute)

    # Carry out the analyses and plot the results.
    for centrality_id, centrality_measure in enumerate(centrality_measures):
        if direction_in_ylabel:
            centrality_col_name = f'{centrality_measure} ({direction})'
        else:
            centrality_col_name = centrality_measure
        df = _compute_centralities(line_graph, centrality_measure, direction, found_values, node_sets, split_attribute,
                                   centrality_col_name)
        _plot_axis(df, centrality_col_name, split_attribute, axes[centrality_id], None, annotate, found_values, plot)

    # Return the figure.
    return utils.return_fig(fig, save_as), df


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


def _compute_centralities(digraph, centrality_measure, direction, found_values, node_sets, split_attribute,
                          centrality_col_name):
    centralities = compute_centralities(digraph, centrality_measure, direction)
    all_attributes = []
    all_centralities = []
    nodes = []
    for value in found_values:
        for node in node_sets[value]:
            all_attributes.append(value)
            all_centralities.append(centralities[node])
            nodes.append(node)
    df = pd.DataFrame(data={centrality_col_name: all_centralities, split_attribute: all_attributes, 'node': nodes})
    return df


def _get_axis(centrality_measures, roles_as_columns, axes, centrality_id, role_id):
    if len(centrality_measures) > 1:
        if roles_as_columns:
            return axes[centrality_id, role_id]
        else:
            return axes[role_id, centrality_id]
    return axes[role_id]


def _plot_axis(df, centrality_col_name, split_attribute, axis, title, annotate, found_values, plot):
    if plot == 'violinplot':
        sns.violinplot(data=df, y=centrality_col_name, x=split_attribute, cut=0, ax=axis)
    else:
        sns.boxplot(data=df, y=centrality_col_name, x=split_attribute, ax=axis)
    if title is not None:
        axis.set_title(title)
    if annotate:
        pairs = list(itt.combinations(found_values, 2))
        annotator = Annotator(axis, pairs, data=df, y=centrality_col_name, x=split_attribute, plot=plot, verbose=True)
        annotator.configure(test='Mann-Whitney', text_format='star', loc='inside', pvalue_format_string='{:.2e}')
        annotator.apply_and_annotate()
