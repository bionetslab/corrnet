import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

import corrnet.utils as utils
from collections import Counter


def plot_neighborhood(multi_digraph, node, edge_types=['in', 'out'], edge_color_info=None, figsize=None,
                      save_as=None, margins=(None, None), font_size=12):
    if figsize is None:
        figsize = (6 * len(edge_types), 6)
    fig, axes = plt.subplots(nrows=1, ncols=len(edge_types), figsize=figsize)
    i = 0
    if 'in' in edge_types:
        in_nbs = list(multi_digraph.predecessors(node))
        if len(edge_types) > 1:
            _plot_neighbors(multi_digraph, node, in_nbs, edge_types[i], edge_color_info, axes[i], margins[i], font_size)
            i += 1
        else:
            _plot_neighbors(multi_digraph, node, in_nbs, edge_types[i], edge_color_info, axes, margins[i], font_size)
    if 'out' in edge_types:
        out_nbs = list(multi_digraph.successors(node))
        if len(edge_types) > 1:
            _plot_neighbors(multi_digraph, node, out_nbs, edge_types[i], edge_color_info, axes[i], margins[i], font_size)
        else:
            _plot_neighbors(multi_digraph, node, out_nbs, edge_types[i], edge_color_info, axes, margins[i], font_size)
    return utils.return_fig(fig, save_as)


def plot_degree_distributions(digraph, figsize=None, save_as=None):
    if figsize is None:
        figsize = (9, 3)
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=figsize)
    in_degrees = dict(digraph.in_degree(weight='weight'))
    out_degrees = dict(digraph.out_degree(weight='weight'))
    total_degrees = {node: in_degrees[node] + out_degrees[node] for node in digraph.nodes()}
    _plot_degree_distribution(total_degrees, 'Total degree', axes[0])
    _plot_degree_distribution(out_degrees, 'Out-degree', axes[1])
    _plot_degree_distribution(in_degrees, 'In-degree', axes[2])
    return utils.return_fig(fig, save_as)


def compute_network_properties(digraph=None, multi_digraph=None, line_graph=None):
    network_types = []
    nums_nodes = []
    nums_edges = []
    nums_wccs = []
    sizes_lwcc = []
    if digraph:
        network_types.append('Digraph')
        _compute_network_properties(digraph, nums_nodes, nums_edges, nums_wccs, sizes_lwcc)
    if multi_digraph:
        network_types.append('Multi-digraph')
        _compute_network_properties(multi_digraph, nums_nodes, nums_edges, nums_wccs, sizes_lwcc)
    if line_graph:
        network_types.append('Directed line graph')
        _compute_network_properties(line_graph, nums_nodes, nums_edges, nums_wccs, sizes_lwcc)
        network_types.append('Undirected line graph')
        _compute_network_properties(nx.Graph(line_graph), nums_nodes, nums_edges, nums_wccs, sizes_lwcc)
    return pd.DataFrame(data={'Network type': network_types,
                              'Num nodes': nums_nodes,
                              'Num edges': nums_edges,
                              'Num WCCs': nums_wccs,
                              'Size LWCC': sizes_lwcc})


def _compute_network_properties(graph, nums_nodes, nums_edges, nums_wccs, sizes_lwcc):
    nums_nodes.append(graph.number_of_nodes())
    nums_edges.append(graph.number_of_edges())
    if graph.is_directed():
        wccs = list(nx.weakly_connected_components(graph))
    else:
        wccs = list(nx.connected_components(graph))
    nums_wccs.append(len(wccs))
    sizes_lwcc.append(len(wccs[0]))


def _plot_degree_distribution(degrees, xlabel, ax):
    degree_counts = Counter(degrees)
    x, y = zip(*degree_counts.items())
    ax.scatter(x, y, marker='.')
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Frequency')
    ax.set_xscale('log')
    ax.set_yscale('log')


def _plot_neighbors(multi_digraph, node, neighbors, edge_type, edge_color_info, ax, margins, font_size):
    h = nx.induced_subgraph(multi_digraph, [node] + neighbors).copy()
    if edge_type == 'in':
        h.remove_edges_from([e for e in h.edges if e[1] != node])
        neighbors = [(e[0], e[2]) for e in h.edges]
    else:
        h.remove_edges_from([e for e in h.edges if e[0] != node])
        neighbors = [(e[1], e[2]) for e in h.edges]
    nodes = [node] + neighbors + list(h.edges)
    if edge_type == 'in':
        edges = [((e[0], e[2]), e) for e in h.edges] + [(e, e[1]) for e in h.edges]
    else:
        edges = [(e, (e[1], e[2])) for e in h.edges] + [(e[0], e) for e in h.edges]
    g = nx.DiGraph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    shells = [[node], list(h.edges), neighbors]
    pos = nx.shell_layout(g, shells, rotate=0)
    edge_color = 'grey'
    node_size = [700] + [500 for _ in neighbors] + [100 for _ in h.edges]
    if edge_color_info:
        edge_color_attribute = edge_color_info[0]
        edge_color_map = edge_color_info[1]
        utils.check_attribute(h, edge_color_attribute)
        node_colors = ['lightskyblue'] + ['lavender' for _ in neighbors] + [edge_color_map[edge[2][edge_color_attribute]]
                                                                         for edge in h.edges(data=True)]
        edge_color = [edge_color_map[edge[2][edge_color_attribute]] for edge in h.edges(data=True)] * 2
        labels = {node: node}
        for t in neighbors:
            labels[t] = t[0]
        for e in h.edges:
            labels[e] = ''
    nx.draw_networkx(g, pos=pos, ax=ax, node_color=node_colors, node_size=node_size, font_size=font_size, margins=margins,
                     edge_color=edge_color, labels=labels)
    ax.set_axis_off()
