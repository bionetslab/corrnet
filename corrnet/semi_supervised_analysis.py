import pandas as pd
import networkx as nx
from networkx.algorithms import node_classification
import corrnet.utils as utils
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def semi_supervised_analysis(line_graph, predict_attribute,
                             methods=['Harmonic function', 'Local and global consistency'], num_cv_runs=100,
                             num_folds=5, figsize=None, save_as=None):
    """Use semi-supervised node classification to assess if correspondence network is predictive of edge label.

    Carries out semi-supervised node classification methods on a line graph representations of a directed correspondence
    network to assess if the network's topology is predictive of a user-selected edge attribute.

    Args:
        line_graph (networkx.DiGraph): Line graph representation of directed correspondence network.
        predict_attribute (str): Node attribute of `line_graph` which should be predicted.
        methods (list of str): Specifies methods to be used for semi-supervised node label prediction. List may contain
            'Harmonic function' and 'Local and global consistency'.
        num_cv_runs (int): Number of cross-validation runs.
        num_folds (int): Number of folds used for cross-validation.
        figsize (tuple or None): Size of the returned figure. If None, the size is determined automatically.
        save_as (str or None): If provided, the returned figure is saved at this path.

    Returns:
        matplotlib.figure.Figure: A figure visualizing the results of the analysis.
    """
    fig, axes = _setup_figure(methods, figsize)
    for i, method_name in enumerate(methods):
        accuracies_for_real_labels = _semi_supervised_analysis(line_graph, predict_attribute, False, num_cv_runs,
                                                               num_folds, method_name)
        accuracies_for_shuffled_labels = _semi_supervised_analysis(line_graph, predict_attribute, True, num_cv_runs,
                                                                   num_folds, method_name)
        df = pd.DataFrame(data={
            'Mean CV accuracy': accuracies_for_real_labels + accuracies_for_shuffled_labels,
            f'{predict_attribute} labels': ['Real' for _ in range(num_cv_runs)] + ['Shuffled' for _ in range(num_cv_runs)]
        })
        sns.histplot(data=df, x='Mean CV accuracy', ax=axes[i], hue=f'{predict_attribute} labels', kde=True)
        axes[i].set_title(f'Classifier: {method_name}')
        axes[i].set_ylabel('Number of CV runs')
    return utils.return_fig(fig, save_as)


def _semi_supervised_analysis(line_graph, predict_attribute, shuffle_labels, num_cv_runs, num_folds, method_name):
    utils.check_attribute(line_graph, predict_attribute)
    node_ids, node_labels = _get_ids_and_labels_of_labeled_nodes(line_graph, predict_attribute)
    mean_cv_accuracies = []
    for _ in range(num_cv_runs):
        if shuffle_labels:
            _shuffle_labels(node_labels)
        folds = _get_folds(node_ids, num_folds)
        accuracies = []
        for test_fold_id, test_fold in enumerate(folds):
            training_fold = [node_id for fold_id in range(num_folds) if fold_id != test_fold_id for node_id in folds[fold_id]]
            undirected_line_graph = nx.Graph(line_graph)
            node_list = list(undirected_line_graph.nodes)
            for node_id in training_fold:
                node = node_list[node_id]
                undirected_line_graph.nodes[node]['label'] = node_labels[node_id]
            predicted_labels = _get_method(method_name)(undirected_line_graph)
            num_correct = 0
            for node_id in test_fold:
                node = node_list[node_id]
                if predicted_labels[node_id] == node_labels[node_id]:
                    num_correct += 1
            accuracies.append(num_correct / len(test_fold))
        mean_cv_accuracies.append(np.mean(accuracies))
    return mean_cv_accuracies


def _get_ids_and_labels_of_labeled_nodes(line_graph, predict_attribute):
    node_ids = []
    node_labels = dict()
    for node_id, node in enumerate(line_graph.nodes(data=True)):
        label = node[1][predict_attribute]
        if pd.isna(label):
            continue
        node_ids.append(node_id)
        node_labels[node_id] = label
    return node_ids, node_labels


def _shuffle_labels(node_labels):
    node_ids = list(node_labels.keys())
    labels = list(node_labels.values())
    np.random.shuffle(labels)
    for i in range(len(node_labels)):
        node_labels[node_ids[i]] = labels[i]


def _get_folds(ids_of_labeled_nodes, num_folds):
    np.random.shuffle(ids_of_labeled_nodes)
    return np.array_split(ids_of_labeled_nodes, num_folds)


def _get_method(method_name):
    method_dict = {
        'Harmonic function': node_classification.harmonic_function,
        'Local and global consistency': node_classification.local_and_global_consistency
    }
    return method_dict[method_name]


def _setup_figure(methods, figsize):
    if figsize is None:
        figsize = (6, 3 * len(methods))
    return plt.subplots(nrows=len(methods), ncols=1, figsize=figsize)
