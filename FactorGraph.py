from Factor import Factor
from copy import copy


class FactorGraph:
    def __init__(self, variables=None, factors=None):
        if variables:
            self.variables = variables
        else:
            self.variables = []

        self.factors = []
        if factors:
            for factor in factors:
                self.add_factor(factor)

    def add_factor(self, factor):
        if set(factor.variables) - set(factor.variables).intersection(set(self.variables)):
            raise ValueError("Factors defined on variable not in the model.")

        self.factors.append(factor)

    def add_variable(self, variable):
        self.variables.append(variable)

    def copy(self):
        return FactorGraph(copy(self.variables), [factor.copy() for factor in self.factors])

    def get_nstate(self, variable):
        for factor in self.factors:
            if variable in factor.variables:
                return factor.values.shape[factor.variables.index(variable)]

    def get_adj_factors(self, variable):
        factor_list = []
        for factor in self.factors:
            if variable in factor.variables:
                factor_list.append(factor)

        return factor_list

    def get_factor(self, name):
        for factor in self.factors:
            if factor.name == name:
                return factor
