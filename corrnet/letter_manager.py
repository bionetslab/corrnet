import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns


class LetterManager:

    def __init__(self, path_letter_data, date_col='date', sender_col='sender', addressee_col='addressee',
                 subject_col=None, provenance_col=None, type_col=None, sep=','):
        self._date_col = date_col
        self._sender_col = sender_col
        self._addressee_col = addressee_col
        self._subject_col = subject_col
        self._provenance_col = provenance_col
        self._type_col = type_col
        self._letter_data = None
        self._bad_letter_data = None
        self._num_letters = None
        self._parse_letter_data(path_letter_data, sep)

    def _parse_letter_data(self, path_letter_data, sep):
        cols = [self._date_col, self._sender_col, self._addressee_col]
        if self._subject_col:
            cols.append(self._subject_col)
        if self._provenance_col:
            cols.append(self._provenance_col)
        if self._type_col:
            cols.append(self._type_col)
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

    def bad_letter_data(self):
        return self._bad_letter_data

    def earliest_date(self):
        return self._letter_data[self._date_col].min()

    def latest_date(self):
        return self._letter_data[self._date_col].max()

    def plot_date_distribution(self, save_as=None, by_type=False):
        fig, ax = plt.subplots()
        if by_type:
            _ = sns.histplot(self._letter_data, x=self._date_col, ax=ax, hue=self._type_col, element='poly')
        else:
            _ = sns.histplot(self._letter_data, x=self._date_col, ax=ax)
        ax.set_ylabel('Number of letters')
        ax.set_xlabel('Date')
        _return_fig(fig, save_as)

    def to_digraph(self, subjects_as_nodes=False, earliest_date=None, latest_date=None, type_filters=None):
        relevant_letters = self._get_relevant_letters(earliest_date, latest_date)
        g = nx.DiGraph()
        g.graph['subjects_as_nodes'] = subjects_as_nodes
        g.graph['sender_col'] = self._sender_col
        g.graph['addressee_col'] = self._addressee_col
        g.graph['subject_col'] = self._subject_col
        g.graph['has_provenance_data'] = (self._provenance_col is not None)
        g.graph['has_edge_type_data'] = (self._type_col is not None)
        for i in range(relevant_letters.shape[0]):
            if self._type_col and type_filters and relevant_letters.loc[i, self._type_col] not in type_filters:
                continue
            self._add_edge_to_digraph(relevant_letters, i, self._sender_col, self._addressee_col, g)
            if subjects_as_nodes:
                self._add_edge_to_digraph(relevant_letters, i, self._subject_col, self._sender_col, g)
        return g

    def _add_edge_to_digraph(self, relevant_letters, i, source_role, target_role, g):
        source = relevant_letters.loc[i, source_role]
        target = relevant_letters.loc[i, target_role]
        if g.has_edge(source, target):
            g[source][target]['weight'] += 1
        else:
            g.add_edge(source, target, weight=1)
        g.nodes[source]['roles'] = g.nodes[source].get('roles', set()).union({source_role})
        g.nodes[target]['roles'] = g.nodes[target].get('roles', set()).union({target_role})
        if self._provenance_col:
            provenance = relevant_letters.loc[i, self._provenance_col]
            g[source][target]['provenances'] = g[source][target].get('provenances', set()).union({provenance})
            g.nodes[source]['provenances'] = g.nodes[source].get('provenances', set()).union({provenance})
            g.nodes[target]['provenances'] = g.nodes[target].get('provenances', set()).union({provenance})
        if self._type_col:
            edge_type = relevant_letters.loc[i, self._type_col]
            g[source][target]['type'] = g[source][target].get('types', set()).union({edge_type})

    def _get_relevant_letters(self, earliest_date, latest_date):
        earliest_date, latest_date = self.init_earliest_and_latest_date(earliest_date, latest_date)
        earliest_date = max(self.earliest_date(), earliest_date)
        latest_date = min(self.latest_date(), latest_date)
        not_too_early = self._letter_data[self._date_col] >= earliest_date
        not_too_late = self._letter_data[self._date_col] <= latest_date
        relevant_letters = self._letter_data[not_too_early & not_too_late]
        relevant_letters.reset_index(inplace=True)
        return relevant_letters

    def init_earliest_and_latest_date(self, earliest_date, latest_date):
        if earliest_date is None:
            earliest_date = self.earliest_date()
        elif isinstance(earliest_date, str):
            earliest_date = pd.to_datetime(earliest_date)
        if latest_date is None:
            latest_date = self.latest_date()
        elif isinstance(latest_date, str):
            latest_date = pd.to_datetime(latest_date)
        return earliest_date, latest_date


def _return_fig(fig, save_as):
    fig.tight_layout()
    if save_as:
        fig.savefig(save_as)
    return fig
