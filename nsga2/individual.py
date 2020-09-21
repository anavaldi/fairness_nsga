class Individual(object):

    def __init__(self, learner):
        self.id = None
        self.rank = None
        self.crowding_distance = None
        self.domination_count = None
        self.dominated_solutions = None
        self.features = None
        self.objectives = None
        self.creation_mode = None
        self.learner_ml = learner

        if self.learner_ml == 'decision_tree':
            self.actual_depth = None
            self.actual_leaves = None

    def __eq__(self, other):
        if isinstance(self, other.__class__):
            return self.features == other.features
        return False

    def dominates_standard(self, other_individual):
        and_condition = True
        or_condition = False
        for first, second in zip(self.objectives, other_individual.objectives):
            and_condition = and_condition and first <= second
            or_condition = or_condition or first < second
        return (and_condition and or_condition)

    def dominates(self, other_individual):
        and_condition = True
        or_condition = False
        eq_condition = True
        for first, second in zip(self.objectives, other_individual.objectives):
            and_condition = and_condition and first <= second
            or_condition = or_condition or first < second
            eq_condition = eq_condition and first == second
            
        if self.learner_ml == 'decision_tree':
            if (eq_condition):
                if ((self.features['max_leaf_nodes'] is None) or (other_individual.features['max_leaf_nodes'] is None)):
                    return (self.actual_leaves < other_individual.actual_leaves)
                else:
                    return ((self.actual_leaves < other_individual.actual_leaves) or
                           ((self.actual_leaves == other_individual.actual_leaves) and (self.features['max_leaf_nodes'] < other_individual.features['max_leaf_nodes'])))
            else:
                return (and_condition and or_condition)
