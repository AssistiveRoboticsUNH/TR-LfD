#!/usr/bin/env python
from itertools import permutations

import networkx as nx

from pgmpy.estimators import StructureEstimator, K2Score
from pgmpy.models import DynamicBayesianNetwork

class HillClimbSearchDBN(StructureEstimator):
    def __init__(self, data, scoring_method=None, **kwargs):
        """
        Class for heuristic hill climb searches for DynamicBayesianModels, to learn
        network structure from data. `estimate` attempts to find a model with optimal score.

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
        if scoring_method is not None:
            self.scoring_method = scoring_method
        else:
            self.scoring_method = K2Score(data, **kwargs)

        super(HillClimbSearchDBN, self).__init__(data, **kwargs)

    def _legal_operations(self, model, tabu_list=[], max_indegree=None):
        """Generates a list of legal (= not in tabu_list) graph modifications
        for a given model, together with their score changes. Possible graph modifications:
        (1) add, (2) remove, or (3) flip a single edge. For details on scoring
        see Koller & Fridman, Probabilistic Graphical Models, Section 18.4.3.3 (page 818).
        If a number `max_indegree` is provided, only modifications that keep the number
        of parents for each node below `max_indegree` are considered."""

        local_score = self.scoring_method.local_score
        nodes = self.state_names.keys()
        potential_new_edges = (set(permutations(nodes, 2)) -
                               set(model.edges()) -
                               set([(Y, X) for (X, Y) in model.edges()]))

        potential_new_edges = set((X, Y) for (X, Y) in potential_new_edges if
                                  (X[1] == Y[1] or (X[1] + 1) == Y[1]))

        for (X, Y) in potential_new_edges:  # (1) add single edge
            X_1, Y_1 = self.get_complementary_edge(X, Y)
            if nx.is_directed_acyclic_graph(nx.DiGraph(list(model.edges()) + [(X, Y), (X_1, Y_1)])):
                operation = ('+', (X, Y))
                if operation not in tabu_list:
                    score_X = self.get_score_add(X, Y, model, max_indegree, local_score)
                    if X_1 != X:
                        score_X_1 = self.get_score_add(X_1, Y_1, model, max_indegree, local_score)
                    else:
                        score_X_1 = 0
                    if score_X is not None and score_X_1 is not None:
                        yield(operation, score_X + score_X_1)

        for (X, Y) in model.edges():  # (2) remove single edge
            X_1, Y_1 = self.get_complementary_edge(X, Y)
            operation = ('-', (X, Y))
            if operation not in tabu_list:
                score_X = self.get_score_remove(X, Y, model, local_score)
                if X != X_1:
                    score_X_1 = self.get_score_remove(X_1, Y_1, model, local_score)
                else:
                    score_X_1 = 0
                yield(operation, score_X + score_X_1)

        flips = set((X, Y) for (X, Y) in model.edges() if X[1] == Y[1])
        for (X, Y) in flips:  # (3) flip single edge
            X_1, Y_1 = self.get_complementary_edge(X, Y)
            new_edges = list(model.edges()) + [(Y, X), (Y_1, X_1)]
            new_edges.remove((X, Y))
            new_edges.remove((X_1, Y_1))
            if nx.is_directed_acyclic_graph(nx.DiGraph(new_edges)):
                operation = ('flip', (X, Y))
                if operation not in tabu_list and ('flip', (Y, X)) not in tabu_list:
                    score_X = self.get_score_flip(X, Y, model, max_indegree, local_score)
                    if X_1 != X:
                        score_X_1 = self.get_score_flip(X_1, Y_1, model, max_indegree, local_score)
                    else:
                        score_X_1 = 0
                    if score_X is not None and score_X_1 is not None:
                        yield(operation, score_X + score_X_1)

    def estimate(self, start=None, tabu_length=0, max_indegree=None):
        """
        Performs local hill climb search to estimates the `BayesianModel` structure
        that has optimal score, according to the scoring method supplied in the constructor.
        Starts at model `start` and proceeds by step-by-step network modifications
        until a local maximum is reached. Only estimates network structure, no parametrization.

        Parameters
        ----------
        start: BayesianModel instance
            The starting point for the local search. By default a completely disconnected network is used.
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
            start = DynamicBayesianNetwork()
            start.add_dynamic_nodes(nodes)
        elif not isinstance(start, DynamicBayesianNetwork):
            raise ValueError("'start' should be a DynamicBayesianModel "
                             "with the same variables as the data set, or 'None'.")

        tabu_list = []
        current_model = start
        iteration_counter = 0
        while True:
            best_score_delta = 0
            best_operation = None

            for operation, score_delta in self._legal_operations(current_model, tabu_list, max_indegree):
                if score_delta > best_score_delta:
                    best_operation = operation
                    best_score_delta = score_delta

            if best_operation is None or best_score_delta < epsilon:
                break
            elif best_operation[0] == '+':
                current_model.add_edge(*best_operation[1])
                tabu_list = ([('-', best_operation[1])] + tabu_list)[:tabu_length]
                if best_operation[1][0][1] == best_operation[1][1][1]:
                    comp_operation = ((best_operation[1][0][0], 1 - best_operation[1][0][1]),
                                      (best_operation[1][1][0], 1 - best_operation[1][1][1]))
                    tabu_list = ([('-', comp_operation)] + tabu_list)[:tabu_length]
            elif best_operation[0] == '-':
                current_model.remove_edge(*best_operation[1])
                tabu_list = ([('+', best_operation[1])] + tabu_list)[:tabu_length]
                if best_operation[1][0][1] == best_operation[1][1][1]:
                    comp_operation = ((best_operation[1][0][0], 1 - best_operation[1][0][1]),
                                      (best_operation[1][1][0], 1 - best_operation[1][1][1]))
                    tabu_list = ([('+', comp_operation)] + tabu_list)[:tabu_length]
            elif best_operation[0] == 'flip':
                X, Y = best_operation[1]
                current_model.remove_edge(X, Y)
                current_model.add_edge(Y, X)
                tabu_list = ([best_operation] + tabu_list)[:tabu_length]
                if best_operation[1][0][1] == best_operation[1][1][1]:
                    comp_operation = ((best_operation[1][0][0], 1 - best_operation[1][0][1]),
                                      (best_operation[1][1][0], 1 - best_operation[1][1][1]))
                    tabu_list = ([('flip', comp_operation)] + tabu_list)[:tabu_length]
            iteration_counter += 1
            if iteration_counter % 100 == 0:
                print iteration_counter
                print current_model.edges()
                print best_score_delta
        return current_model

    def get_complementary_edge(self, X, Y):
        if X[1] == Y[1]:
            X_1 = (X[0], 1 - X[1])
            Y_1 = (Y[0], 1 - Y[1])
        else:
            X_1 = X
            Y_1 = Y
        return X_1, Y_1

    def get_score_add(self, X, Y, model, max_indegree, local_score):
        old_parents = model.get_parents(Y)
        new_parents = list(old_parents) + [X]
        if max_indegree is None or len(new_parents) <= max_indegree:
            return local_score(Y, list(new_parents)) - local_score(Y, list(old_parents))
        return None

    def get_score_remove(self, X, Y, model, local_score):
        old_parents = model.get_parents(Y)
        new_parents = list(old_parents)[:]
        new_parents.remove(X)
        return local_score(Y, list(new_parents)) - local_score(Y, list(old_parents))

    def get_score_flip(self, X, Y, model, max_indegree, local_score):
        old_X_parents = list(model.get_parents(X))
        old_Y_parents = list(model.get_parents(Y))
        new_X_parents = old_X_parents + [Y]
        new_Y_parents = old_Y_parents[:]
        new_Y_parents.remove(X)
        if max_indegree is None or len(new_X_parents) <= max_indegree:
            return (local_score(X, new_X_parents) +
                    local_score(Y, new_Y_parents) -
                    local_score(X, old_X_parents) -
                    local_score(Y, old_Y_parents))
