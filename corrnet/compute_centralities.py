import networkx as nx
import json
import corrnet.utils as utils


def compute_centralities(digraph, centrality_measure='PageRank centrality', direction='in', normalize=False,
                         as_sorted_list=False, save_as=None):
    """

    Args:
        digraph (networkx.DiGraph): Directed graph.
        centrality_measure (str): Select centrality measure. Options: 'PageRank centrality', 'Harmonic centrality',
            'Betweenness centrality', 'Degree centrality'.
        direction (str): Select 'in' or 'out'.
        normalize (bool): If True, centralities are normalized.
        as_sorted_list (bool): If True, return centralities as sorted list.
        save_as (str or None): Path to JSON file where results should be stored.

    Returns:
        Centralities for all nodes, either as dictionary (if as_sorted_list=False) or as sorted list.
    """
    centrality_fun = _get_centrality_fun(centrality_measure)
    centralities = None
    if direction == 'in':
        centralities = _compute_centralities(digraph, centrality_fun, normalize)
    if direction == 'out':
        centralities = _compute_centralities(digraph.reverse(copy=False), centrality_fun, normalize)
    if as_sorted_list:
        centralities = list(centralities.items())
        centralities.sort(key=lambda t: t[1], reverse=True)
    if save_as:
        with open(save_as, mode='w') as fp:
            json.dump(centralities, fp, indent='\t', sort_keys=True)
    return centralities


def _get_centrality_fun(centrality_measure):
    centrality_fun_dict = {
        'PageRank centrality': nx.pagerank_scipy,
        'Harmonic centrality': nx.centrality.harmonic_centrality,
        'Betweenness centrality': nx.centrality.betweenness_centrality,
        'Degree centrality': nx.centrality.in_degree_centrality,
        'Closeness centrality': nx.centrality.closeness_centrality
    }
    return centrality_fun_dict[centrality_measure]


def _compute_centralities(digraph, centrality_fun, normalize):
    centralities = centrality_fun(digraph)
    if normalize:
        centralities = utils.normalized(centralities)
    return centralities
