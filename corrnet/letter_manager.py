import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns
import corrnet.utils as utils
import warnings


class LetterManager:

    def __init__(self, path_letter_data, date_col='date', sender_col='sender', addressee_col='addressee',
                 attribute_cols=[], sep=',', show_warnings=False):
        """Constructs a LetterManager object.

        Args:
            path_letter_data (str): Path to CSV file containing letter meta-data.
            date_col (str): Name of date column in input file.
            sender_col (str): Name of sender column in input file.
            addressee_col (str): Name of addressee column in input file.
            attribute_cols (list of str): Names of further attribute columns in input to be loaded.
            sep (char): Seperator in input file.
            show_warnings (bool): If True, warnings are printed to stdout.
        """
        self._date_col = date_col
        self._sender_col = sender_col
        self._addressee_col = addressee_col
        self._attribute_cols = attribute_cols
        self._letter_data = None
        self._bad_letter_data = None
        self._num_letters = None
        self._parse_letter_data(path_letter_data, sep, show_warnings)

    def letter_data(self):
        """Returns letter data.

        Returns:
            pandas.DataFrame: Data frame containing parsed letter data.
        """
        return self._letter_data

    def bad_letter_data(self):
        """Returns invalid letter data.

        Returns:
            pandas.DataFrame: Data frame containing letter records in input file with invalid dates.
        """
        return self._bad_letter_data

    def earliest_date(self, filter_by=None):
        """Returns earliest date of a letter record matching the filter criterion.

        Args:
            filter_by (tuple of str or None): None or an attribute-value pair to be used as positive filter when finding
            the earliest date. If None, all records are considered.

        Returns:
            pandas._libs.tslibs.timestamps.Timestamp: The earliest date of a letter matching the filter criterion.
        """
        if filter_by and filter_by[0] in self._attribute_cols:
            return self._letter_data[self._letter_data[filter_by[0]] == filter_by[1]][self._date_col].min()
        return self._letter_data[self._date_col].min()

    def latest_date(self, filter_by=None):
        """Returns latest date of a letter record matching the filter criterion.

        Args:
            filter_by (tuple of str or None): None or an attribute-value pair to be used as positive filter when finding
            the latest date. If None, all records are considered.

        Returns:
            pandas._libs.tslibs.timestamps.Timestamp: The latest date of a letter matching the filter criterion.
        """
        if filter_by and filter_by[0] in self._attribute_cols:
            return self._letter_data[self._letter_data[filter_by[0]] == filter_by[1]][self._date_col].max()
        return self._letter_data[self._date_col].max()

    def plot_date_distribution(self, save_as=None, split_attribute=None):
        """Plots date distribution of the letter records.

        Args:
            save_as (str or None): If provided, the returned figure is saved at this path.
            split_attribute (str or None): If provided, separate distributions are visualized for all values of
            `split_attribute`.

        Returns:
            matplotlib.figure.Figure: A figure visualizing date distribution.
        """
        fig, ax = plt.subplots()
        if split_attribute and split_attribute in self._attribute_cols:
            _ = sns.histplot(self._letter_data, x=self._date_col, ax=ax, element='poly', hue=split_attribute)
        else:
            _ = sns.histplot(self._letter_data, x=self._date_col, ax=ax)
        ax.set_ylabel('Number of letters')
        ax.set_xlabel('Date')
        return utils.return_fig(fig, save_as)

    def construct_graphs(self, earliest_date=None, latest_date=None, build_digraph=True, build_multi_digraph=True,
                         build_line_graph=True, filter_by=None):
        """Constructs graph representation of the letter records.

        Args:
            earliest_date (str or None): If not None, all records with earlier dates are ignored when constructing the
                graphs.
            latest_date (str or None): If not None, all records with later dates are ignored when constructing the
                graphs.
            build_digraph (bool): If True, a digraph representation is constructed.
            build_multi_digraph (bool): If True, a multi-digraph representation is constructed.
            build_line_graph (bool): If True, a line graph representation is constructed.
            filter_by (tuple of str or None): None or an attribute-value pair to be used as positive filter when
            constructing the graphs. If None, all records are considered.

        Returns:
            networkx.DiGraph or None: Digraph representation of the letter records (or None if `build_digraph` is
                False).
            networkx.MultiDiGraph or None: Multi-digraph representation of the letter records (None if
                `build_multi_digraph` is False).
            networkx.DiGraph or None: Line graph representation of the letter records (or None if `build_line_graph` is
                False).
        """
        relevant_letters = self._get_relevant_letters(earliest_date, latest_date, filter_by)
        build_multi_digraph = build_multi_digraph or build_line_graph
        digraph = multi_digraph = line_graph = None
        if build_digraph:
            digraph = nx.DiGraph()
        if build_multi_digraph:
            multi_digraph = nx.MultiDiGraph()
            self._add_graph_attributes(multi_digraph)
        for i in range(relevant_letters.shape[0]):
            if build_digraph:
                self._add_edge_to_digraph(relevant_letters, i, digraph)
            if build_multi_digraph:
                self._add_edge_to_multi_digraph(relevant_letters, i, multi_digraph)
        if build_line_graph:
            line_graph = self._build_line_graph(multi_digraph)
        return digraph, multi_digraph, line_graph

    def _parse_letter_data(self, path_letter_data, sep, show_warnings):
        cols = [self._date_col, self._sender_col, self._addressee_col] + self._attribute_cols
        self._letter_data = pd.read_csv(path_letter_data, usecols=cols, sep=sep)
        self._check_dates(show_warnings)
        self._remove_duplicates(show_warnings)
        self._num_letters = self._letter_data.shape[0]

    def _check_dates(self, show_warnings):
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
            self._letter_data.reset_index(inplace=True, drop=True)
            if show_warnings:
                warnings.warn(f'Found and removed {len(idx_bad_dates)} records with bad dates.')
        self._letter_data[self._date_col] = pd.to_datetime(self._letter_data[self._date_col])

    def _remove_duplicates(self, show_warnings):
        keep = self._letter_data.duplicated().apply(lambda b: not b)
        num_duplicates = self._letter_data.shape[0] - keep.aggregate('sum')
        if num_duplicates > 0 and show_warnings:
            warnings.warn(f'Found and removed {num_duplicates} duplicate records.')
        self._letter_data = self._letter_data[keep]
        self._letter_data.reset_index(drop=True, inplace=True)

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

    def _get_relevant_letters(self, earliest_date, latest_date, filter_by):
        earliest_date, latest_date = self._init_earliest_and_latest_date(earliest_date, latest_date)
        earliest_date = max(self.earliest_date(), earliest_date)
        latest_date = min(self.latest_date(), latest_date)
        not_too_early = self._letter_data[self._date_col] >= earliest_date
        not_too_late = self._letter_data[self._date_col] <= latest_date
        relevant_letters = self._letter_data[not_too_early & not_too_late]
        relevant_letters.reset_index(inplace=True, drop=True)
        if filter_by and filter_by[0] in self._attribute_cols:
            relevant_letters = relevant_letters[relevant_letters[filter_by[0]] == filter_by[1]]
            relevant_letters.reset_index(inplace=True, drop=True)
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
        line_graph = nx.DiGraph(nx.line_graph(multi_digraph))
        self._add_graph_attributes(line_graph)
        for node in line_graph.nodes(data=True):
            for attribute in self._attribute_cols + [self._date_col]:
                node[1][attribute] = multi_digraph.edges[node[0]][attribute]
        return line_graph
