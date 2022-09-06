import networkx as nx
import json
import corrnet.utils as utils


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
        centralities = utils.normalized(centralities)
    return centralities
