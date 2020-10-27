import numpy as np
from FactorGraph import FactorGraph
from Factor import Factor


def ith_factor_name(i):
    return "F" + str(i)


def ijth_variable_name(i, j):
    return "V(" + str(i) + "," + str(j) + ")"


def ith_variable_name(i):
    return "V" + str(i)


class CompleteFactorGraph(FactorGraph):
    def __init__(self, nfactors):
        super(CompleteFactorGraph, self).__init__()

        factorNvariables = {i: [] for i in range(nfactors)}
        for i in range(nfactors):
            for j in range(i + 1, nfactors):
                self.add_variable(ijth_variable_name(i, j))
                factorNvariables[i].append(ijth_variable_name(i, j))
                factorNvariables[j].append(ijth_variable_name(i, j))

        for i in range(nfactors):
            cardinality = [2 for j in range(nfactors - 1)]
            factor_value = np.random.random(cardinality)
            self.add_factor(
                Factor(ith_factor_name(i), factorNvariables[i], cardinality, factor_value)
            )


class ThreeRegularFactorGraph(FactorGraph):
    def __init__(self, nfactors, T=1.0):
        super(ThreeRegularFactorGraph, self).__init__()

        factorNvariables = {i: [] for i in range(nfactors)}
        for i in range(nfactors):
            if i == 0:
                j = nfactors - 1
            else:
                j = i - 1

            self.add_variable(ijth_variable_name(i, j))
            factorNvariables[i].append(ijth_variable_name(i, j))
            factorNvariables[j].append(ijth_variable_name(i, j))

            if i < nfactors / 2:
                _, j = divmod(i + (nfactors / 2), nfactors)
                j = int(j)
                self.add_variable(ijth_variable_name(i, j))
                factorNvariables[i].append(ijth_variable_name(i, j))
                factorNvariables[j].append(ijth_variable_name(i, j))

        for i in range(nfactors):
            cardinality = [2 for j in range(len(factorNvariables[i]))]
            factor_value = np.exp(np.random.normal(0.0, T, cardinality))
            self.add_factor(
                Factor(ith_factor_name(i), factorNvariables[i], cardinality, factor_value)
            )
