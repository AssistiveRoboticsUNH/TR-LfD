#!/usr/bin/env python3

from collections import defaultdict

import networkx as nx
import pandas as pd
import numpy as np
import os

from pgmpy.models import BayesianModel
from pgmpy.models.MarkovModel import MarkovModel


class IntervalTemporalBayesianNetwork(BayesianModel):
    """
    Base class for interval temporal bayesian network model. (ITBN)

    A model stores nodes and edges with conditional probability
    distribution (cpd) and other attributes.

    models hold directed edges.  Self loops are not allowed neither
    multiple (parallel) edges.

    Nodes can be any hashable python object.

    Edges are represented as links between nodes.
    """
    observation_node_marker = "obs_"
    temporal_node_marker = "tm_"
    start_time_marker = "_s"
    end_time_marker = "_e"

    # Allen's Temporal Relations
    BEFORE = 1
    BEFORE_INV = 2
    MEETS = 3
    MEETS_INV = 4
    OVERLAPS = 5
    OVERLAPS_INV = 6
    DURING = 7
    DURING_INV = 8
    STARTS = 9
    STARTS_INV = 10
    FINISHES = 11
    FINISHES_INV = 12
    EQUAL = 13
    AIG_MAP = {1: 'b', 2: 'bi',
               3: 'm', 4: 'mi',
               5: 'o', 6: 'oi',
               7: 'd', 8: 'di',
               9: 's', 10: 'si',
               11: 'f', 12: 'fi',
               13: 'e'}

    def __init__(self, ebunch=None):
        super(IntervalTemporalBayesianNetwork, self).__init__(ebunch)
        self.relation_map = None
        self.event_nodes = None
        self.interval_relation_map = self.load_interval_relation_map()

    def add_edge(self, u, v, **kwargs):
        """
        Add an edge between u and v.

        The nodes u and v will be automatically added if they are
        not already in the graph

        Parameters
        ----------
        u,v : nodes
              Nodes can be any hashable python object.
        """
        super(IntervalTemporalBayesianNetwork, self).add_edge(u, v, **kwargs)

    def remove_node(self, node):
        """
        Remove node from the model.

        Removing a node also removes all the associated edges, all associated temporal nodes,
        removes the CPD of the node and marginalizes the CPDs of it's children.

        Parameters
        ----------
        node : node
            Node which is to be removed from the model.

        Returns
        -------
        None
        """
        affected_nodes = [v for u, v in self.edges() if u == node]

        for affected_node in affected_nodes:
            node_cpd = self.get_cpds(node=affected_node)
            if node_cpd:
                node_cpd.marginalize([node], inplace=True)

        if self.get_cpds(node=node):
            self.remove_cpds(node)
        super(BayesianModel, self).remove_node(node)

        for affected_node in affected_nodes:
            if affected_node.startswith(self.temporal_node_marker):
                self.remove_node(affected_node)

    def remove_nodes_from(self, nodes):
        """
        Remove multiple nodes from the model.

        Removing a node also removes all the associated edges, removes the CPD
        of the node and marginalizes the CPDs of it's children.

        Parameters
        ----------
        nodes : list, set (iterable)
            Nodes which are to be removed from the model.

        Returns
        -------
        None
        """
        for node in nodes:
            self.remove_node(node)

    def check_model(self):
        """
        Check the model for various errors. This method checks for the following
        errors.

        * Checks if the sum of the probabilities for each state is equal to 1 (tol=0.01).
        * Checks if the CPDs associated with nodes are consistent with their parents.

        Returns
        -------
        check: boolean
            True if all the checks are passed
        """
        return super(IntervalTemporalBayesianNetwork, self).check_model()

    def to_markov_model(self):
        """
        Converts bayesian model to markov model. The markov model created would
        be the moral graph of the bayesian model.
        """
        moral_graph = self.moralize()
        mm = MarkovModel(moral_graph.edges())
        mm.add_nodes_from(moral_graph.nodes())
        mm.add_factors(*[cpd.to_factor() for cpd in self.cpds])

        return mm

    def to_junction_tree(self):
        """
        Creates a junction tree (or clique tree) for a given bayesian model.

        For converting a Bayesian Model into a Clique tree, first it is converted
        into a Markov one.

        For a given markov model (H) a junction tree (G) is a graph
        1. where each node in G corresponds to a maximal clique in H
        2. each sepset in G separates the variables strictly on one side of the
        edge to other.
        """
        mm = self.to_markov_model()
        return mm.to_junction_tree()

    def fit(self, data, estimator=None, state_names=None, complete_samples_only=True, **kwargs):
        """
        Estimates the CPD for each variable based on a given data set.

        Parameters
        ----------
        data: pandas DataFrame object
            DataFrame object with column names identical to the variable names of the network.
            (If some values in the data are missing the data cells should be set to `numpy.NaN`.
            Note that pandas converts each column containing `numpy.NaN`s to dtype `float`.)

        estimator: Estimator class
            One of:
            - MaximumLikelihoodEstimator (default)
            - BayesianEstimator: In this case, pass 'prior_type' and either 'pseudo_counts'
                or 'equivalent_sample_size' as additional keyword arguments.
                See `BayesianEstimator.get_parameters()` for usage.

        state_names: dict (optional)
            A dict indicating, for each variable, the discrete set of states
            that the variable can take. If unspecified, the observed values
            in the data set are taken to be the only possible states.

        complete_samples_only: bool (default `True`)
            Specifies how to deal with missing data, if present. If set to `True` all rows
            that contain `np.Nan` somewhere are ignored. If `False` then, for each variable,
            every row where neither the variable nor its parents are `np.NaN` is used.
        """

        if state_names is None:
            state_names = []
        from pgmpy.estimators import MaximumLikelihoodEstimator, BaseEstimator

        if estimator is None:
            estimator = MaximumLikelihoodEstimator
        else:
            if not issubclass(estimator, BaseEstimator):
                raise TypeError("Estimator object should be a valid pgmpy estimator.")

        _estimator = estimator(self, data, state_names=state_names,
                               complete_samples_only=complete_samples_only)
        cpds_list = _estimator.get_parameters(**kwargs)
        self.add_cpds(*cpds_list)

    def predict(self, data):
        """
        Predicts states of all the missing variables.

        Parameters
        ----------
        data : pandas DataFrame object
            A DataFrame object with column names same as the variables in the model.
        """
        from pgmpy.inference import VariableElimination

        if set(data.columns) == set(self.nodes()):
            raise ValueError("No variable missing in data. Nothing to predict")

        elif set(data.columns) - set(self.nodes()):
            raise ValueError("Data has variables which are not in the model")

        missing_variables = list(set(self.nodes()) - set(data.columns))
        pred_values = defaultdict(list)

        # Send state_names dict from one of the estimated CPDs to the inference class.
        state_names = dict()
        for cpd in self.get_cpds():
            for key, value in cpd.state_names.items():
                state_names[key] = value
        model_inference = VariableElimination(self, state_names=state_names)
        for index, data_point in data.iterrows():
            states_dict = model_inference.map_query(variables=missing_variables,
                                                    evidence=data_point.to_dict())
            for k, v in states_dict.items():
                pred_values[k].append(v)
        return pd.DataFrame(pred_values, index=data.index)

    def predict_probability(self, data):
        """
        Predicts probabilities of all states of the missing variables.

        Parameters
        ----------
        data : pandas DataFrame object
            A DataFrame object with column names same as the variables in the model.
        """
        from pgmpy.inference import VariableElimination

        if set(data.columns) == set(self.nodes()):
            raise ValueError("No variable missing in data. Nothing to predict")

        elif set(data.columns) - set(self.nodes()):
            raise ValueError("Data has variables which are not in the model")

        missing_variables = list(set(self.nodes()) - set(data.columns))
        pred_values = defaultdict(list)

        model_inference = VariableElimination(self)
        for index, data_point in data.iterrows():
            states_dict = model_inference.query(variables=missing_variables,
                                                evidence=data_point.to_dict())
            for k, v in states_dict.items():
                for l in range(len(v.values)):
                    state = self.get_cpds(k).state_names[k][l]
                    pred_values[k + '_' + str(state)].append(v.values[l])
        return pd.DataFrame(pred_values, index=data.index)

    def is_iequivalent(self, model):
        """
        Checks whether the given model is I-equivalent

        Two graphs G1 and G2 are said to be I-equivalent if they have same skeleton
        and have same set of immoralities.

        Note: For same skeleton different names of nodes can work but for immoralities
        names of nodes must be same

        Parameters
        ----------
        model : A Bayesian model object, for which you want to check I-equivalence

        Returns
        --------
        boolean : True if both are I-equivalent, False otherwise
        """
        if not isinstance(model, IntervalTemporalBayesianNetwork):
            raise TypeError('model must be an instance of ITBN Model')
        skeleton = nx.algorithms.isomorphism.GraphMatcher(self.to_undirected(),
                                                          model.to_undirected())
        if skeleton.is_isomorphic() and self.get_immoralities() == model.get_immoralities():
            return True
        return False

    def learn_temporal_relationships(self, data):
        relation_map = dict()
        if self.event_nodes is None:
            self.event_nodes = set(
                [node for node in self.nodes() if not node.startswith(self.temporal_node_marker)])
        for node_a in self.event_nodes:
            for node_b in self.event_nodes:
                relation_set = set()
                if not node_a == node_b:
                    for sample in data.itertuples():
                        if (getattr(sample, node_a + self.start_time_marker) >= 0 and
                                getattr(sample, node_b + self.start_time_marker) >= 0):
                            relation = self.calculate_relationship(sample, node_a, node_b)
                            if relation % 2 == 0 or relation == self.EQUAL:
                                relation_set.add(relation)
                                data.at[getattr(sample, 'Index'), self.temporal_node_marker +
                                        node_a + "_" + node_b] = relation
                relation_map[(node_a, node_b)] = sorted(relation_set)
        self.add_nodes_from(list(data.columns.values))
        self.relation_map = relation_map

    def calculate_relationship(self, sample, node_a, node_b):
        start_a = getattr(sample, node_a + self.start_time_marker)
        end_a = getattr(sample, node_a + self.end_time_marker)
        start_b = getattr(sample, node_b + self.start_time_marker)
        end_b = getattr(sample, node_b + self.end_time_marker)
        temp_distance = (np.sign(start_b - start_a), np.sign(end_b - end_a),
                         np.sign(start_b - end_a), np.sign(end_b - start_a))
        return self.interval_relation_map[temp_distance]

    def load_interval_relation_map(self):
        interval_relation_map = dict()
        interval_relation_map[(-1., -1., -1., -1.)] = self.BEFORE
        interval_relation_map[(1., 1., 1., 1.)] = self.BEFORE_INV
        interval_relation_map[(1., -1., -1., 1.)] = self.DURING
        interval_relation_map[(-1., 1., -1., 1.)] = self.DURING_INV
        interval_relation_map[(-1., -1., -1., 1.)] = self.OVERLAPS
        interval_relation_map[(1., 1., -1., 1.)] = self.OVERLAPS_INV
        interval_relation_map[(-1., -1., -1., 0.)] = self.MEETS
        interval_relation_map[(1., 1., 0., 1.)] = self.MEETS_INV
        interval_relation_map[(0., -1., -1., 1.)] = self.STARTS
        interval_relation_map[(0., 1., -1., 1.)] = self.STARTS_INV
        interval_relation_map[(1., 0., -1., 1.)] = self.FINISHES
        interval_relation_map[(-1., 0., -1., 1.)] = self.FINISHES_INV
        interval_relation_map[(0., 0., -1., 1.)] = self.EQUAL
        return interval_relation_map

    def add_temporal_nodes(self):
        current_edges = list(self.edges())
        for edge in current_edges:
            relation_set = self.relation_map[edge]
            if len(relation_set) > 0:
                temporal_node = self.temporal_node_marker + edge[0] + "_" + edge[1]
                if temporal_node not in self.nodes():
                    self.add_node(temporal_node)
                    self.add_edge(edge[0], temporal_node)
                    self.add_edge(edge[1], temporal_node)

    def draw_to_file(self, file_path, include_obs=False):
        drawn_edges = set()
        output = "strict digraph {\n"
        if self.event_nodes is None:
            self.event_nodes = set(
                [node for node in self.nodes() if (not node.startswith(self.temporal_node_marker)
                                                   and not node.startswith(self.observation_node_marker))])
        for event in self.event_nodes:
            output += event + " [weight=None]\n"
        if include_obs:
            for node in self.nodes():
                if node.startswith(self.observation_node_marker):
                    output += node + " [weight=None, style=dotted]\n"
        for edge in self.edges():
            if (edge[1].startswith(self.temporal_node_marker) and
                    edge[1] not in drawn_edges):
                drawn_edges.add(edge[1])
                edge_nodes = edge[1].replace(self.temporal_node_marker, "").split("_")
                relations_str = ",".join(self.AIG_MAP[r] for r in
                                         self.relation_map[(edge_nodes[0], edge_nodes[1])])
                output += edge_nodes[0] + " -> " + edge_nodes[1] + " [weight=None, label=\" " + \
                          relations_str + " \"]\n"
            elif include_obs and edge[1].startswith(self.observation_node_marker):
                output += edge[0] + " -> " + edge[1] + " [weight=None, style=dotted]\n"
        output += "}"
        dot_file = file_path.replace('.png', '.dot')
        with open(dot_file, "w") as output_file:
            output_file.write(output)
        os.system('dot ' + dot_file + ' -Tpng -o ' + file_path)

    def add_nodes_from(self, nodes, weights=None):
        nodes = [node for node in nodes if not node.endswith(self.start_time_marker)
                 and not node.endswith(self.end_time_marker)]
        super(IntervalTemporalBayesianNetwork, self).add_nodes_from(nodes, weights)

    def event_edges(self):
        edges = [edge for edge in super(IntervalTemporalBayesianNetwork, self).edges() if
                 not edge[1].startswith(self.temporal_node_marker)]
        return edges

    def load_interval_relation_transitivity_table(self):
        ir_transitivity_table = dict()
        # ########################################################################    BEFORE
        ir_transitivity_table[(self.BEFORE, self.BEFORE)] = [self.BEFORE]
        ir_transitivity_table[(self.BEFORE, self.BEFORE_INV)] = None
        ir_transitivity_table[(self.BEFORE, self.DURING)] = [self.BEFORE,
                                                             self.OVERLAPS,
                                                             self.MEETS,
                                                             self.DURING,
                                                             self.STARTS]
        ir_transitivity_table[(self.BEFORE, self.DURING_INV)] = [self.BEFORE]
        ir_transitivity_table[(self.BEFORE, self.OVERLAPS)] = [self.BEFORE]
        ir_transitivity_table[(self.BEFORE, self.OVERLAPS_INV)] = [self.BEFORE,
                                                                   self.OVERLAPS,
                                                                   self.MEETS,
                                                                   self.DURING,
                                                                   self.STARTS]
        ir_transitivity_table[(self.BEFORE, self.MEETS)] = [self.BEFORE]
        ir_transitivity_table[(self.BEFORE, self.MEETS_INV)] = [self.BEFORE,
                                                                self.OVERLAPS,
                                                                self.MEETS,
                                                                self.DURING,
                                                                self.STARTS]
        ir_transitivity_table[(self.BEFORE, self.STARTS)] = [self.BEFORE]
        ir_transitivity_table[(self.BEFORE, self.STARTS_INV)] = [self.BEFORE]
        ir_transitivity_table[(self.BEFORE, self.FINISHES)] = [self.BEFORE,
                                                               self.OVERLAPS,
                                                               self.MEETS,
                                                               self.DURING,
                                                               self.STARTS]
        ir_transitivity_table[(self.BEFORE, self.FINISHES_INV)] = [self.BEFORE]
        ir_transitivity_table[(self.BEFORE, self.EQUAL)] = [self.BEFORE]

        # ########################################################################    BEFORE_INV
        ir_transitivity_table[(self.BEFORE_INV, self.BEFORE)] = None
        ir_transitivity_table[(self.BEFORE_INV, self.BEFORE_INV)] = [self.BEFORE_INV]
        ir_transitivity_table[(self.BEFORE_INV, self.DURING)] = [self.BEFORE_INV,
                                                                 self.OVERLAPS_INV,
                                                                 self.MEETS_INV,
                                                                 self.DURING,
                                                                 self.FINISHES]
        ir_transitivity_table[(self.BEFORE_INV, self.DURING_INV)] = [self.BEFORE_INV]
        ir_transitivity_table[(self.BEFORE_INV, self.OVERLAPS)] = [self.BEFORE_INV,
                                                                   self.OVERLAPS_INV,
                                                                   self.MEETS_INV,
                                                                   self.DURING,
                                                                   self.FINISHES]
        ir_transitivity_table[(self.BEFORE_INV, self.OVERLAPS_INV)] = [self.BEFORE_INV]
        ir_transitivity_table[(self.BEFORE_INV, self.MEETS)] = [self.BEFORE_INV,
                                                                self.OVERLAPS_INV,
                                                                self.MEETS_INV,
                                                                self.DURING,
                                                                self.FINISHES]
        ir_transitivity_table[(self.BEFORE_INV, self.MEETS_INV)] = [self.BEFORE_INV]
        ir_transitivity_table[(self.BEFORE_INV, self.STARTS)] = [self.BEFORE_INV,
                                                                 self.OVERLAPS_INV,
                                                                 self.MEETS_INV,
                                                                 self.DURING,
                                                                 self.FINISHES]
        ir_transitivity_table[(self.BEFORE_INV, self.STARTS_INV)] = [self.BEFORE_INV]
        ir_transitivity_table[(self.BEFORE_INV, self.FINISHES)] = [self.BEFORE_INV]
        ir_transitivity_table[(self.BEFORE_INV, self.FINISHES_INV)] = [self.BEFORE_INV]
        ir_transitivity_table[(self.BEFORE_INV, self.EQUAL)] = [self.BEFORE_INV]

        # ########################################################################    DURING
        ir_transitivity_table[(self.DURING, self.BEFORE)] = [self.BEFORE]
        ir_transitivity_table[(self.DURING, self.BEFORE_INV)] = [self.BEFORE_INV]
        ir_transitivity_table[(self.DURING, self.DURING)] = [self.DURING]
        ir_transitivity_table[(self.DURING, self.DURING_INV)] = None
        ir_transitivity_table[(self.DURING, self.OVERLAPS)] = [self.BEFORE,
                                                               self.OVERLAPS,
                                                               self.MEETS,
                                                               self.DURING,
                                                               self.STARTS]
        ir_transitivity_table[(self.DURING, self.OVERLAPS_INV)] = [self.BEFORE_INV,
                                                                   self.OVERLAPS_INV,
                                                                   self.MEETS_INV,
                                                                   self.DURING,
                                                                   self.FINISHES]
        ir_transitivity_table[(self.DURING, self.MEETS)] = [self.BEFORE]
        ir_transitivity_table[(self.DURING, self.MEETS_INV)] = [self.BEFORE_INV]
        ir_transitivity_table[(self.DURING, self.STARTS)] = [self.DURING]
        ir_transitivity_table[(self.DURING, self.STARTS_INV)] = [self.BEFORE_INV,
                                                                 self.OVERLAPS_INV,
                                                                 self.MEETS_INV,
                                                                 self.DURING,
                                                                 self.FINISHES]
        ir_transitivity_table[(self.DURING, self.FINISHES)] = [self.DURING]
        ir_transitivity_table[(self.DURING, self.FINISHES_INV)] = [self.BEFORE,
                                                                   self.OVERLAPS,
                                                                   self.MEETS,
                                                                   self.DURING,
                                                                   self.STARTS]
        ir_transitivity_table[(self.DURING, self.EQUAL)] = [self.DURING]

        # ########################################################################    DURING_INV
        ir_transitivity_table[(self.DURING_INV, self.BEFORE)] = [self.BEFORE,
                                                                 self.OVERLAPS,
                                                                 self.MEETS,
                                                                 self.DURING_INV,
                                                                 self.FINISHES_INV]
        ir_transitivity_table[(self.DURING_INV, self.BEFORE_INV)] = [self.BEFORE_INV,
                                                                     self.OVERLAPS_INV,
                                                                     self.MEETS_INV,
                                                                     self.DURING_INV,
                                                                     self.STARTS_INV]
        ir_transitivity_table[(self.DURING_INV, self.DURING)] = [self.OVERLAPS,
                                                                 self.OVERLAPS_INV,
                                                                 self.DURING,
                                                                 self.STARTS,
                                                                 self.FINISHES,
                                                                 self.DURING_INV,
                                                                 self.STARTS_INV,
                                                                 self.FINISHES_INV,
                                                                 self.EQUAL]
        ir_transitivity_table[(self.DURING_INV, self.DURING_INV)] = [self.DURING_INV]
        ir_transitivity_table[(self.DURING_INV, self.OVERLAPS)] = [self.OVERLAPS,
                                                                   self.DURING_INV,
                                                                   self.FINISHES_INV]
        ir_transitivity_table[(self.DURING_INV, self.OVERLAPS_INV)] = [self.OVERLAPS_INV,
                                                                       self.DURING_INV,
                                                                       self.STARTS_INV]
        ir_transitivity_table[(self.DURING_INV, self.MEETS)] = [self.OVERLAPS,
                                                                self.DURING_INV,
                                                                self.FINISHES_INV]
        ir_transitivity_table[(self.DURING_INV, self.MEETS_INV)] = [self.OVERLAPS_INV,
                                                                    self.DURING_INV,
                                                                    self.STARTS_INV]
        ir_transitivity_table[(self.DURING_INV, self.STARTS)] = [self.DURING_INV,
                                                                 self.FINISHES_INV,
                                                                 self.OVERLAPS]
        ir_transitivity_table[(self.DURING_INV, self.STARTS_INV)] = [self.DURING_INV]
        ir_transitivity_table[(self.DURING_INV, self.FINISHES)] = [self.DURING_INV,
                                                                   self.STARTS_INV,
                                                                   self.OVERLAPS_INV]
        ir_transitivity_table[(self.DURING_INV, self.FINISHES_INV)] = [self.DURING_INV]
        ir_transitivity_table[(self.DURING_INV, self.EQUAL)] = [self.DURING_INV]

        # ########################################################################    OVERLAPS
        ir_transitivity_table[(self.OVERLAPS, self.BEFORE)] = [self.BEFORE]
        ir_transitivity_table[(self.OVERLAPS, self.BEFORE_INV)] = [self.BEFORE_INV,
                                                                   self.OVERLAPS_INV,
                                                                   self.DURING_INV,
                                                                   self.MEETS_INV,
                                                                   self.STARTS_INV]
        ir_transitivity_table[(self.OVERLAPS, self.DURING)] = [self.OVERLAPS,
                                                               self.DURING,
                                                               self.STARTS]
        ir_transitivity_table[(self.OVERLAPS, self.DURING_INV)] = [self.BEFORE,
                                                                   self.OVERLAPS,
                                                                   self.MEETS,
                                                                   self.DURING_INV,
                                                                   self.FINISHES_INV]
        ir_transitivity_table[(self.OVERLAPS, self.OVERLAPS)] = [self.BEFORE,
                                                                 self.OVERLAPS,
                                                                 self.MEETS]
        ir_transitivity_table[(self.OVERLAPS, self.OVERLAPS_INV)] = [self.OVERLAPS,
                                                                     self.OVERLAPS_INV,
                                                                     self.DURING,
                                                                     self.STARTS,
                                                                     self.FINISHES,
                                                                     self.DURING_INV,
                                                                     self.STARTS_INV,
                                                                     self.FINISHES_INV,
                                                                     self.EQUAL]
        ir_transitivity_table[(self.OVERLAPS, self.MEETS)] = [self.BEFORE]
        ir_transitivity_table[(self.OVERLAPS, self.MEETS_INV)] = [self.OVERLAPS_INV,
                                                                  self.DURING_INV,
                                                                  self.STARTS_INV]
        ir_transitivity_table[(self.OVERLAPS, self.STARTS)] = [self.OVERLAPS]
        ir_transitivity_table[(self.OVERLAPS, self.STARTS_INV)] = [self.DURING_INV,
                                                                   self.FINISHES_INV,
                                                                   self.OVERLAPS]
        ir_transitivity_table[(self.OVERLAPS, self.FINISHES)] = [self.DURING,
                                                                 self.STARTS,
                                                                 self.OVERLAPS]
        ir_transitivity_table[(self.OVERLAPS, self.FINISHES_INV)] = [self.BEFORE,
                                                                     self.OVERLAPS,
                                                                     self.MEETS]
        ir_transitivity_table[(self.OVERLAPS, self.EQUAL)] = [self.OVERLAPS]

        # ########################################################################    OVERLAPS_INV
        ir_transitivity_table[(self.OVERLAPS_INV, self.BEFORE)] = [self.BEFORE,
                                                                   self.OVERLAPS,
                                                                   self.MEETS,
                                                                   self.DURING_INV,
                                                                   self.FINISHES_INV]
        ir_transitivity_table[(self.OVERLAPS_INV, self.BEFORE_INV)] = [self.BEFORE_INV]
        ir_transitivity_table[(self.OVERLAPS_INV, self.DURING)] = [self.OVERLAPS_INV,
                                                                   self.DURING,
                                                                   self.FINISHES]
        ir_transitivity_table[(self.OVERLAPS_INV, self.DURING_INV)] = [self.BEFORE_INV,
                                                                       self.OVERLAPS_INV,
                                                                       self.MEETS_INV,
                                                                       self.DURING_INV,
                                                                       self.STARTS_INV]
        ir_transitivity_table[(self.OVERLAPS_INV, self.OVERLAPS)] = [self.OVERLAPS,
                                                                     self.OVERLAPS_INV,
                                                                     self.DURING,
                                                                     self.STARTS,
                                                                     self.FINISHES,
                                                                     self.DURING_INV,
                                                                     self.STARTS_INV,
                                                                     self.FINISHES_INV,
                                                                     self.EQUAL]
        ir_transitivity_table[(self.OVERLAPS_INV, self.OVERLAPS_INV)] = [self.BEFORE_INV,
                                                                         self.OVERLAPS_INV,
                                                                         self.MEETS_INV]
        ir_transitivity_table[(self.OVERLAPS_INV, self.MEETS)] = [self.OVERLAPS,
                                                                  self.DURING_INV,
                                                                  self.FINISHES_INV]
        ir_transitivity_table[(self.OVERLAPS_INV, self.MEETS_INV)] = [self.BEFORE_INV]
        ir_transitivity_table[(self.OVERLAPS_INV, self.STARTS)] = [self.OVERLAPS_INV,
                                                                   self.DURING,
                                                                   self.FINISHES]
        ir_transitivity_table[(self.OVERLAPS_INV, self.STARTS_INV)] = [self.OVERLAPS_INV,
                                                                       self.BEFORE_INV,
                                                                       self.MEETS_INV]
        ir_transitivity_table[(self.OVERLAPS_INV, self.FINISHES)] = [self.OVERLAPS_INV]
        ir_transitivity_table[(self.OVERLAPS_INV, self.FINISHES_INV)] = [self.OVERLAPS_INV,
                                                                         self.DURING_INV,
                                                                         self.STARTS_INV]
        ir_transitivity_table[(self.OVERLAPS_INV, self.EQUAL)] = [self.OVERLAPS_INV]

        # ########################################################################    MEETS
        ir_transitivity_table[(self.MEETS, self.BEFORE)] = [self.BEFORE]
        ir_transitivity_table[(self.MEETS, self.BEFORE_INV)] = [self.BEFORE_INV,
                                                                self.OVERLAPS_INV,
                                                                self.MEETS_INV,
                                                                self.DURING_INV,
                                                                self.STARTS_INV]
        ir_transitivity_table[(self.MEETS, self.DURING)] = [self.OVERLAPS,
                                                            self.DURING,
                                                            self.STARTS]
        ir_transitivity_table[(self.MEETS, self.DURING_INV)] = [self.BEFORE]
        ir_transitivity_table[(self.MEETS, self.OVERLAPS)] = [self.BEFORE]
        ir_transitivity_table[(self.MEETS, self.OVERLAPS_INV)] = [self.OVERLAPS,
                                                                  self.DURING,
                                                                  self.STARTS]
        ir_transitivity_table[(self.MEETS, self.MEETS)] = [self.BEFORE]
        ir_transitivity_table[(self.MEETS, self.MEETS_INV)] = [self.FINISHES,
                                                               self.FINISHES_INV,
                                                               self.EQUAL]
        ir_transitivity_table[(self.MEETS, self.STARTS)] = [self.MEETS]
        ir_transitivity_table[(self.MEETS, self.STARTS_INV)] = [self.MEETS]
        ir_transitivity_table[(self.MEETS, self.FINISHES)] = [self.DURING,
                                                              self.STARTS,
                                                              self.OVERLAPS]
        ir_transitivity_table[(self.MEETS, self.FINISHES_INV)] = [self.BEFORE]
        ir_transitivity_table[(self.MEETS, self.EQUAL)] = [self.MEETS]

        # ########################################################################    MEETS_INV
        ir_transitivity_table[(self.MEETS_INV, self.BEFORE)] = [self.BEFORE,
                                                                self.OVERLAPS,
                                                                self.MEETS,
                                                                self.DURING_INV,
                                                                self.FINISHES_INV]
        ir_transitivity_table[(self.MEETS_INV, self.BEFORE_INV)] = [self.BEFORE_INV]
        ir_transitivity_table[(self.MEETS_INV, self.DURING)] = [self.OVERLAPS_INV,
                                                                self.DURING,
                                                                self.FINISHES]
        ir_transitivity_table[(self.MEETS_INV, self.DURING_INV)] = [self.BEFORE_INV]
        ir_transitivity_table[(self.MEETS_INV, self.OVERLAPS)] = [self.OVERLAPS_INV,
                                                                  self.DURING,
                                                                  self.FINISHES]
        ir_transitivity_table[(self.MEETS_INV, self.OVERLAPS_INV)] = [self.BEFORE_INV]
        ir_transitivity_table[(self.MEETS_INV, self.MEETS)] = [self.STARTS,
                                                               self.STARTS_INV,
                                                               self.EQUAL]
        ir_transitivity_table[(self.MEETS_INV, self.MEETS_INV)] = [self.BEFORE_INV]
        ir_transitivity_table[(self.MEETS_INV, self.STARTS)] = [self.DURING,
                                                                self.FINISHES,
                                                                self.OVERLAPS_INV]
        ir_transitivity_table[(self.MEETS_INV, self.STARTS_INV)] = [self.BEFORE_INV]
        ir_transitivity_table[(self.MEETS_INV, self.FINISHES)] = [self.MEETS_INV]
        ir_transitivity_table[(self.MEETS_INV, self.FINISHES_INV)] = [self.MEETS_INV]
        ir_transitivity_table[(self.MEETS_INV, self.EQUAL)] = [self.MEETS_INV]

        # ########################################################################    STARTS
        ir_transitivity_table[(self.STARTS, self.BEFORE)] = [self.BEFORE]
        ir_transitivity_table[(self.STARTS, self.BEFORE_INV)] = [self.BEFORE_INV]
        ir_transitivity_table[(self.STARTS, self.DURING)] = [self.DURING]
        ir_transitivity_table[(self.STARTS, self.DURING_INV)] = [self.BEFORE,
                                                                 self.OVERLAPS,
                                                                 self.MEETS,
                                                                 self.DURING_INV,
                                                                 self.FINISHES_INV]
        ir_transitivity_table[(self.STARTS, self.OVERLAPS)] = [self.BEFORE,
                                                               self.OVERLAPS,
                                                               self.MEETS]
        ir_transitivity_table[(self.STARTS, self.OVERLAPS_INV)] = [self.OVERLAPS_INV,
                                                                   self.DURING,
                                                                   self.FINISHES]
        ir_transitivity_table[(self.STARTS, self.MEETS)] = [self.BEFORE]
        ir_transitivity_table[(self.STARTS, self.MEETS_INV)] = [self.MEETS_INV]
        ir_transitivity_table[(self.STARTS, self.STARTS)] = [self.STARTS]
        ir_transitivity_table[(self.STARTS, self.STARTS_INV)] = [self.STARTS,
                                                                 self.STARTS_INV,
                                                                 self.EQUAL]
        ir_transitivity_table[(self.STARTS, self.FINISHES)] = [self.DURING]
        ir_transitivity_table[(self.STARTS, self.FINISHES_INV)] = [self.BEFORE,
                                                                   self.MEETS,
                                                                   self.OVERLAPS]
        ir_transitivity_table[(self.STARTS, self.EQUAL)] = [self.STARTS]

        # ########################################################################    STARTS_INV
        ir_transitivity_table[(self.STARTS_INV, self.BEFORE)] = [self.BEFORE,
                                                                 self.OVERLAPS,
                                                                 self.MEETS,
                                                                 self.DURING_INV,
                                                                 self.FINISHES_INV]
        ir_transitivity_table[(self.STARTS_INV, self.BEFORE_INV)] = [self.BEFORE_INV]
        ir_transitivity_table[(self.STARTS_INV, self.DURING)] = [self.OVERLAPS_INV,
                                                                 self.DURING,
                                                                 self.FINISHES]
        ir_transitivity_table[(self.STARTS_INV, self.DURING_INV)] = [self.DURING_INV]
        ir_transitivity_table[(self.STARTS_INV, self.OVERLAPS)] = [self.OVERLAPS,
                                                                   self.DURING_INV,
                                                                   self.FINISHES_INV]
        ir_transitivity_table[(self.STARTS_INV, self.OVERLAPS_INV)] = [self.OVERLAPS_INV]
        ir_transitivity_table[(self.STARTS_INV, self.MEETS)] = [self.OVERLAPS,
                                                                self.DURING_INV,
                                                                self.FINISHES_INV]
        ir_transitivity_table[(self.STARTS_INV, self.MEETS_INV)] = [self.MEETS_INV]
        ir_transitivity_table[(self.STARTS_INV, self.STARTS)] = [self.STARTS,
                                                                 self.STARTS_INV,
                                                                 self.EQUAL]
        ir_transitivity_table[(self.STARTS_INV, self.STARTS_INV)] = [self.STARTS_INV]
        ir_transitivity_table[(self.STARTS_INV, self.FINISHES)] = [self.OVERLAPS_INV]
        ir_transitivity_table[(self.STARTS_INV, self.FINISHES_INV)] = [self.DURING_INV]
        ir_transitivity_table[(self.STARTS_INV, self.EQUAL)] = [self.STARTS_INV]

        # ########################################################################    FINISHES
        ir_transitivity_table[(self.FINISHES, self.BEFORE)] = [self.BEFORE]
        ir_transitivity_table[(self.FINISHES, self.BEFORE_INV)] = [self.BEFORE_INV]
        ir_transitivity_table[(self.FINISHES, self.DURING)] = [self.DURING]
        ir_transitivity_table[(self.FINISHES, self.DURING_INV)] = [self.BEFORE_INV,
                                                                   self.OVERLAPS_INV,
                                                                   self.MEETS_INV,
                                                                   self.DURING_INV,
                                                                   self.STARTS_INV]
        ir_transitivity_table[(self.FINISHES, self.OVERLAPS)] = [self.OVERLAPS,
                                                                 self.DURING,
                                                                 self.STARTS]
        ir_transitivity_table[(self.FINISHES, self.OVERLAPS_INV)] = [self.BEFORE,
                                                                     self.OVERLAPS_INV,
                                                                     self.MEETS]
        ir_transitivity_table[(self.FINISHES, self.MEETS)] = [self.MEETS]
        ir_transitivity_table[(self.FINISHES, self.MEETS_INV)] = [self.BEFORE_INV]
        ir_transitivity_table[(self.FINISHES, self.STARTS)] = [self.DURING]
        ir_transitivity_table[(self.FINISHES, self.STARTS_INV)] = [self.BEFORE_INV,
                                                                   self.OVERLAPS_INV,
                                                                   self.MEETS_INV]
        ir_transitivity_table[(self.FINISHES, self.FINISHES)] = [self.FINISHES]
        ir_transitivity_table[(self.FINISHES, self.FINISHES_INV)] = [self.FINISHES,
                                                                     self.FINISHES_INV]
        ir_transitivity_table[(self.FINISHES, self.EQUAL)] = [self.FINISHES]

        # ########################################################################    FINISHES_INV
        ir_transitivity_table[(self.FINISHES_INV, self.BEFORE)] = [self.BEFORE]
        ir_transitivity_table[(self.FINISHES_INV, self.BEFORE_INV)] = [self.BEFORE_INV,
                                                                       self.OVERLAPS_INV,
                                                                       self.MEETS_INV,
                                                                       self.DURING_INV,
                                                                       self.STARTS_INV]
        ir_transitivity_table[(self.FINISHES_INV, self.DURING)] = [self.OVERLAPS,
                                                                   self.DURING,
                                                                   self.STARTS]
        ir_transitivity_table[(self.FINISHES_INV, self.DURING_INV)] = [self.DURING_INV]
        ir_transitivity_table[(self.FINISHES_INV, self.OVERLAPS)] = [self.OVERLAPS]
        ir_transitivity_table[(self.FINISHES_INV, self.OVERLAPS_INV)] = [self.OVERLAPS_INV,
                                                                         self.DURING_INV,
                                                                         self.STARTS_INV]
        ir_transitivity_table[(self.FINISHES_INV, self.MEETS)] = [self.MEETS]
        ir_transitivity_table[(self.FINISHES_INV, self.MEETS_INV)] = [self.STARTS_INV,
                                                                      self.OVERLAPS_INV,
                                                                      self.DURING_INV]
        ir_transitivity_table[(self.FINISHES_INV, self.STARTS)] = [self.OVERLAPS]
        ir_transitivity_table[(self.FINISHES_INV, self.STARTS_INV)] = [self.DURING_INV]
        ir_transitivity_table[(self.FINISHES_INV, self.FINISHES)] = [self.FINISHES,
                                                                     self.FINISHES_INV,
                                                                     self.EQUAL]
        ir_transitivity_table[(self.FINISHES_INV, self.FINISHES_INV)] = [self.FINISHES_INV]
        ir_transitivity_table[(self.FINISHES_INV, self.EQUAL)] = [self.FINISHES_INV]

        # ########################################################################    EQUAL
        ir_transitivity_table[(self.EQUAL, self.BEFORE)] = [self.BEFORE]
        ir_transitivity_table[(self.EQUAL, self.BEFORE_INV)] = [self.BEFORE_INV]
        ir_transitivity_table[(self.EQUAL, self.DURING)] = [self.DURING]
        ir_transitivity_table[(self.EQUAL, self.DURING_INV)] = [self.DURING_INV]
        ir_transitivity_table[(self.EQUAL, self.OVERLAPS)] = [self.OVERLAPS]
        ir_transitivity_table[(self.EQUAL, self.OVERLAPS_INV)] = [self.OVERLAPS_INV]
        ir_transitivity_table[(self.EQUAL, self.MEETS)] = [self.MEETS]
        ir_transitivity_table[(self.EQUAL, self.MEETS_INV)] = [self.MEETS_INV]
        ir_transitivity_table[(self.EQUAL, self.STARTS)] = [self.STARTS]
        ir_transitivity_table[(self.EQUAL, self.STARTS_INV)] = [self.STARTS_INV]
        ir_transitivity_table[(self.EQUAL, self.FINISHES)] = [self.FINISHES]
        ir_transitivity_table[(self.EQUAL, self.FINISHES_INV)] = [self.FINISHES_INV]
        ir_transitivity_table[(self.EQUAL, self.EQUAL)] = [self.EQUAL]

        return ir_transitivity_table

    def learn_temporal_relationships_from_cpds(self):
        self.relation_map = dict()
        for cpd in self.cpds:
            for key, value in cpd.state_names.items():
                if key.startswith(IntervalTemporalBayesianNetwork.temporal_node_marker):
                    nodes = key.replace(IntervalTemporalBayesianNetwork.temporal_node_marker, '').split('_')
                    new_key = (nodes[0], nodes[1])
                    relations = list()
                    for element in value:
                        if element != 0:
                            relations.append(element)
                    self.relation_map[new_key] = sorted(relations)
