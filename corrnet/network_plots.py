import networkx as nx
import matplotlib.pyplot as plt
import corrnet.utils as utils


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
    utils.return_fig(fig, save_as)


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
