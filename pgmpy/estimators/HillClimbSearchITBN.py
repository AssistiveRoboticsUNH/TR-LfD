#!/usr/bin/env python
from itertools import permutations

import networkx as nx

from pgmpy.estimators import HillClimbSearch
from pgmpy.models import IntervalTemporalBayesianNetwork as ITBN


class HillClimbSearchITBN(HillClimbSearch):
    def __init__(self, data, **kwargs):
        """
        Class for heuristic hill climb searches for ITBN models, to learn
        network structure from data. `estimate` attempts to find a model with optimal score.
        Adapted to work with the ITBN model proposed by Zhang, et al.

        Parameters
        ----------
        data: pandas DataFrame object
            datafame object where each column represents one variable.
            (If some values in the data are missing the data cells should be set to `numpy.NaN`.
            Note that pandas converts each column containing `numpy.NaN`s to dtype `float`.)

        scoring_method: Instance of a `StructureScore`-subclass (`K2Score` is used as default)
            An instance of `K2Score`, `BdeuScore`, or `BicScore`.
            This score is optimized during structure estimation by the `estimate`-method.

        state_names: dict (optional)
            A dict indicating, for each variable, the discrete set of states (or values)
            that the variable can take. If unspecified, the observed values in the data set
            are taken to be the only possible states.

        complete_samples_only: bool (optional, default `True`)
            Specifies how to deal with missing data, if present. If set to `True` all rows
            that contain `np.Nan` somewhere are ignored. If `False` then, for each variable,
            every row where neither the variable nor its parents are `np.NaN` is used.
            This sets the behavior of the `state_count`-method.
        """
        super(HillClimbSearchITBN, self).__init__(data, **kwargs)

    def _legal_operations(self, model, tabu_list=[], max_indegree=None):
        """Generates a list of legal (= not in tabu_list) graph modifications
        for a given model, together with their score changes. Possible graph modifications:
        (1) add, (2) remove, or (3) flip a single edge. For details on scoring
        see Koller & Fridman, Probabilistic Graphical Models, Section 18.4.3.3 (page 818).
        If a number `max_indegree` is provided, only modifications that keep the number
        of parents for each node below `max_indegree` are considered."""

        local_score = self.scoring_method.local_score
        nodes = model.event_nodes
        potential_new_edges = (set(permutations(nodes, 2)) -
                               set(model.edges()) -
                               set([(Y, X) for (X, Y) in model.edges()]) -
                               set([X for X, Y in model.relation_map.items() if len(Y) == 0]))

        for (X, Y) in potential_new_edges:  # (1) add single edge
            if nx.is_directed_acyclic_graph(nx.DiGraph(list(model.edges()) + [(X, Y)])):
                if self.valid_temporal_relations([(X, Y)], model):
                    operation = ('+', (X, Y))
                    if operation not in tabu_list:
                        old_parents = list(model.get_parents(Y))
                        new_parents = old_parents + [X]
                        if max_indegree is None or len(new_parents) <= max_indegree:
                            temporal_node_parents = [X, Y]
                            temporal_node = ITBN.temporal_node_marker + X + "_" + Y
                            score_delta = (local_score(Y, new_parents) -
                                           local_score(Y, old_parents) +
                                           local_score(temporal_node, temporal_node_parents))
                            yield(operation, score_delta)

        for (X, Y) in model.event_edges():  # (2) remove single edge
            operation = ('-', (X, Y))
            if operation not in tabu_list:
                old_parents = list(model.get_parents(Y))
                new_parents = old_parents[:]
                new_parents.remove(X)
                temporal_node_parents = [X, Y]
                temporal_node = ITBN.temporal_node_marker + X + "_" + Y
                score_delta = (local_score(Y, new_parents) -
                               local_score(Y, old_parents) -
                               local_score(temporal_node, temporal_node_parents))
                yield(operation, score_delta)

        for (X, Y) in model.event_edges():  # (3) flip single edge
            if len(model.relation_map[(Y, X)]) > 0:
                new_edges = list(model.edges()) + [(Y, X)]
                new_edges.remove((X, Y))
                if nx.is_directed_acyclic_graph(nx.DiGraph(new_edges)):
                    if self.valid_temporal_relations([(X, Y)], model):
                        operation = ('flip', (X, Y))
                        if operation not in tabu_list and ('flip', (Y, X)) not in tabu_list:
                            old_X_parents = list(model.get_parents(X))
                            old_Y_parents = list(model.get_parents(Y))
                            new_X_parents = old_X_parents + [Y]
                            new_Y_parents = old_Y_parents[:]
                            new_Y_parents.remove(X)
                            if max_indegree is None or len(new_X_parents) <= max_indegree:
                                temporal_node_parents = [Y, X]
                                temporal_node = ITBN.temporal_node_marker + Y + "_" + X
                                old_temp_node_parents = [X, Y]
                                old_temp_node = ITBN.temporal_node_marker + X + "_" + Y
                                score_delta = (local_score(X, new_X_parents) +
                                               local_score(Y, new_Y_parents) -
                                               local_score(X, old_X_parents) -
                                               local_score(Y, old_Y_parents) +
                                               local_score(temporal_node, temporal_node_parents) -
                                               local_score(old_temp_node, old_temp_node_parents))
                                yield(operation, score_delta)

    def estimate(self, start=None, tabu_length=0, max_indegree=None):
        """
        Performs local hill climb search to estimates the `BayesianModel` structure
        that has optimal score, according to the scoring method supplied in the constructor.
        Starts at model `start` and proceeds by step-by-step network modifications
        until a local maximum is reached. Only estimates network structure, no parametrization.

        Parameters
        ----------
        start: BayesianModel instance
            The starting point for the local search. By default a completely disconnected
            network is used.
        tabu_length: int
            If provided, the last `tabu_length` graph modifications cannot be reversed
            during the search procedure. This serves to enforce a wider exploration
            of the search space. Default value: 100.
        max_indegree: int or None
            If provided and unequal None, the procedure only searches among models
            where all nodes have at most `max_indegree` parents. Defaults to None.

        Returns
        -------
        model: `BayesianModel` instance
            A `BayesianModel` at a (local) score maximum.

        Examples
        --------
        >>> import pandas as pd
        >>> import numpy as np
        >>> from pgmpy.estimators import HillClimbSearch, BicScore
        >>> # create data sample with 9 random variables:
        ... data = pd.DataFrame(np.random.randint(0, 5, size=(5000, 9)), columns=list('ABCDEFGHI'))
        >>> # add 10th dependent variable
        ... data['J'] = data['A'] * data['B']
        >>> est = HillClimbSearch(data, scoring_method=BicScore(data))
        >>> best_model = est.estimate()
        >>> sorted(best_model.nodes())
        ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
        >>> best_model.edges()
        [('B', 'J'), ('A', 'J')]
        >>> # search a model with restriction on the number of parents:
        >>> est.estimate(max_indegree=1).edges()
        [('J', 'A'), ('B', 'J')]
        """
        epsilon = 1e-8
        nodes = self.state_names.keys()
        if start is None:
            start = ITBN()
            start.add_nodes_from(nodes)
        elif (not isinstance(start, ITBN) or
              not set(start.nodes()) == set(nodes)):
            raise ValueError("'start' should be a IntervalTemporalBayesianNetwork with the same "
                             "variables as the data set, or 'None'.")

        tabu_list = []
        current_model = start

        while True:
            best_score_delta = 0
            best_operation = None

            for operation, score_delta in self._legal_operations(current_model,
                                                                 tabu_list, max_indegree):
                if score_delta > best_score_delta:
                    best_operation = operation
                    best_score_delta = score_delta
                # # the following condition gives preference to inverse relations in order to facilitate the
                # # interpretation of the final model and the inference process
                # if (best_operation is not None and abs(score_delta - best_score_delta) < epsilon and
                #         operation[1][0] == best_operation[1][1] and operation[1][1] == best_operation[1][0]):
                #     if len(current_model.relation_map[operation[1]]) > 0:
                #         # inverse relations have even identifiers
                #         if current_model.relation_map[operation[1]][0] % 2 == 0:
                #             best_operation = operation

            if best_operation is None or best_score_delta < epsilon:
                break
            elif best_operation[0] == '+':
                current_model.add_edge(*best_operation[1])
                temporal_node = (ITBN.temporal_node_marker + best_operation[1][0] +
                                 "_" + best_operation[1][1])
                current_model.add_edge(best_operation[1][0], temporal_node)
                current_model.add_edge(best_operation[1][1], temporal_node)
                tabu_list = ([('-', best_operation[1])] + tabu_list)[:tabu_length]
            elif best_operation[0] == '-':
                current_model.remove_edge(*best_operation[1])
                temporal_node = (ITBN.temporal_node_marker + best_operation[1][0] +
                                 "_" + best_operation[1][1])
                current_model.remove_edge(best_operation[1][0], temporal_node)
                current_model.remove_edge(best_operation[1][1], temporal_node)
                tabu_list = ([('+', best_operation[1])] + tabu_list)[:tabu_length]
            elif best_operation[0] == 'flip':
                X, Y = best_operation[1]
                current_model.remove_edge(X, Y)
                current_model.add_edge(Y, X)
                old_temp_node = (ITBN.temporal_node_marker + best_operation[1][0] +
                                 "_" + best_operation[1][1])
                current_model.remove_edge(best_operation[1][0], old_temp_node)
                current_model.remove_edge(best_operation[1][1], old_temp_node)
                temporal_node = (ITBN.temporal_node_marker + best_operation[1][1] +
                                 "_" + best_operation[1][0])
                current_model.add_edge(best_operation[1][0], temporal_node)
                current_model.add_edge(best_operation[1][1], temporal_node)
                tabu_list = ([best_operation] + tabu_list)[:tabu_length]

        return current_model

    def valid_temporal_relations(self, new_edge, model):
        return True
        edges = list(model.edges()) + [new_edge]
        triangles = [tri for tri in nx.enumerate_all_cliques(nx.Graph(edges)) if len(tri) == 3]
        if len(triangles) < 1:
            return True
        return False

    def get_triangles(self, edges, new_edge):
        return None
