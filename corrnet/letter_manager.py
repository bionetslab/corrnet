import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns
import corrnet.utils as utils


class LetterManager:

    def __init__(self, path_letter_data, date_col='date', sender_col='sender', addressee_col='addressee',
                 attribute_cols=[], sep=','):
        self._date_col = date_col
        self._sender_col = sender_col
        self._addressee_col = addressee_col
        self._attribute_cols = attribute_cols
        self._letter_data = None
        self._bad_letter_data = None
        self._num_letters = None
        self._parse_letter_data(path_letter_data, sep)

    def letter_data(self):
        return self._letter_data

    def bad_letter_data(self):
        return self._bad_letter_data

    def earliest_date(self):
        return self._letter_data[self._date_col].min()

    def latest_date(self):
        return self._letter_data[self._date_col].max()

    def plot_date_distribution(self, save_as=None, by_type=False):
        fig, ax = plt.subplots()
        if by_type:
            _ = sns.histplot(self._letter_data, x=self._date_col, ax=ax, element='poly')
        else:
            _ = sns.histplot(self._letter_data, x=self._date_col, ax=ax)
        ax.set_ylabel('Number of letters')
        ax.set_xlabel('Date')
        utils.return_fig(fig, save_as)

    def construct_graphs(self, earliest_date=None, latest_date=None, build_digraph=True, build_multi_digraph=True,
                         build_line_graph=True, filter_by=None):
        relevant_letters = self._get_relevant_letters(earliest_date, latest_date)
        build_multi_digraph = build_multi_digraph or build_line_graph
        digraph = multi_digraph = line_graph = None
        if build_digraph:
            digraph = nx.DiGraph()
        if build_multi_digraph:
            multi_digraph = nx.MultiDiGraph()
            self._add_graph_attributes(multi_digraph)
        for i in range(relevant_letters.shape[0]):
            if filter_by and filter_by[0] in self._attribute_cols:
                if relevant_letters.loc[i, filter_by[0]] not in filter_by[1]:
                    continue
            if build_digraph:
                self._add_edge_to_digraph(relevant_letters, i, digraph)
            if build_multi_digraph:
                self._add_edge_to_multi_digraph(relevant_letters, i, multi_digraph)
        if build_line_graph:
            line_graph = self._build_line_graph(multi_digraph)
        return digraph, multi_digraph, line_graph

    def _parse_letter_data(self, path_letter_data, sep):
        cols = [self._date_col, self._sender_col, self._addressee_col] + self._attribute_cols
        self._letter_data = pd.read_csv(path_letter_data, usecols=cols, sep=sep)
        idx_bad_dates = []
        for i in range(self._letter_data.shape[0]):
            try:
                date = pd.to_datetime(self._letter_data.loc[i, self._date_col])
                if pd.isnull(date):
                    idx_bad_dates.append(i)
            except ValueError:
                idx_bad_dates.append(i)
        if len(idx_bad_dates) > 0:
            self._bad_letter_data = self._letter_data.loc[idx_bad_dates].copy()
            self._letter_data.drop(index=idx_bad_dates, inplace=True)
            self._letter_data.reset_index(inplace=True)
        self._letter_data[self._date_col] = pd.to_datetime(self._letter_data[self._date_col])
        self._num_letters = self._letter_data.shape[0]

    def _add_edge_to_digraph(self, relevant_letters, i, digraph):
        source = relevant_letters.loc[i, self._sender_col]
        target = relevant_letters.loc[i, self._addressee_col]
        if digraph.has_edge(source, target):
            digraph[source][target]['weight'] += 1
        else:
            digraph.add_edge(source, target, weight=1)

    def _add_edge_to_multi_digraph(self, relevant_letters, i, multi_digraph):
        source = relevant_letters.loc[i, self._sender_col]
        target = relevant_letters.loc[i, self._addressee_col]
        key = multi_digraph.add_edge(source, target)
        multi_digraph[source][target][key][self._date_col] = relevant_letters.loc[i, self._date_col]
        for attribute_col in self._attribute_cols:
            multi_digraph[source][target][key][attribute_col] = relevant_letters.loc[i, attribute_col]

    def _get_relevant_letters(self, earliest_date, latest_date):
        earliest_date, latest_date = self._init_earliest_and_latest_date(earliest_date, latest_date)
        earliest_date = max(self.earliest_date(), earliest_date)
        latest_date = min(self.latest_date(), latest_date)
        not_too_early = self._letter_data[self._date_col] >= earliest_date
        not_too_late = self._letter_data[self._date_col] <= latest_date
        relevant_letters = self._letter_data[not_too_early & not_too_late]
        relevant_letters.reset_index(inplace=True)
        return relevant_letters

    def _init_earliest_and_latest_date(self, earliest_date, latest_date):
        if earliest_date is None:
            earliest_date = self.earliest_date()
        elif isinstance(earliest_date, str):
            earliest_date = pd.to_datetime(earliest_date)
        if latest_date is None:
            latest_date = self.latest_date()
        elif isinstance(latest_date, str):
            latest_date = pd.to_datetime(latest_date)
        return earliest_date, latest_date

    def _add_graph_attributes(self, graph):
        graph.graph['sender_attribute_name'] = self._sender_col
        graph.graph['addressee_attribute_name'] = self._addressee_col
        graph.graph['attribute_names'] = [self._date_col]
        if len(self._attribute_cols) > 0:
            graph.graph['attribute_names'] += self._attribute_cols

    def _build_line_graph(self, multi_digraph):
        line_graph = nx.Graph(nx.line_graph(multi_digraph))
        self._add_graph_attributes(line_graph)
        for node in line_graph.nodes(data=True):
            for attribute in self._attribute_cols + [self._date_col]:
                node[1][attribute] = multi_digraph.edges[node[0]][attribute]
        return line_graph
