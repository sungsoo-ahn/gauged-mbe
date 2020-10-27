import numpy as np
from functools import reduce
from copy import copy
import sys


class Factor:
    def __init__(self, name, variables, cardinality, values):
        self.name = name
        self.variables = variables

        self.cardinality = cardinality
        self.values = np.array(values.reshape(cardinality))

    def get_cardinality(self, variables):
        return {var: self.cardinality[self.variables.index(var)] for var in variables}

    def copy(self):
        return Factor(
            copy(self.name), copy(self.variables), copy(self.cardinality), copy(self.values),
        )

    def full_like(self, value):
        return Factor(
            copy(self.name),
            copy(self.variables),
            copy(self.cardinality),
            np.full_like(self.values, value),
        )

    def power(self, w, inplace=True):
        phi = self if inplace else self.copy()
        try:
            phi.values = np.power(phi.values, w)
        except:
            print(phi.values)
            print(w)

        if not inplace:
            return phi

    def abs(self, inplace=True):
        phi = self if inplace else self.copy()
        phi.values = np.abs(phi.values)
        if not inplace:
            return phi

    def transpose(self, variables, inplace=True):
        phi = self if inplace else self.copy()
        phi.values = phi.values.transpose([phi.variables.index(var) for var in variables])
        if not inplace:
            return phi

    def invT(self, inplace=True):
        phi = self if inplace else self.copy()
        phi.values = np.transpose(np.linalg.inv(phi.values), (1, 0))
        if not inplace:
            return phi

    def marginalize(self, variables, inplace=True):
        phi = self if inplace else self.copy()

        for var in variables:
            if var not in phi.variables:
                raise ValueError("{var} not in scope.".format(var=var))

        var_indices = [phi.variables.index(var) for var in variables]
        index_to_keep = sorted(set(range(len(self.variables))) - set(var_indices))
        phi.variables = [phi.variables[index] for index in index_to_keep]
        phi.cardinality = [phi.cardinality[index] for index in index_to_keep]

        phi.values = np.sum(phi.values, axis=tuple(var_indices))
        if not inplace:
            return phi

    def condition(self, variables, values, inplace=True):
        phi = self if inplace else self.copy()

        for var in variables:
            if var not in phi.variables:
                raise ValueError("{var} not in scope.".format(var=var))

        var_indices_to_del = []
        slice_ = [slice(None)] * len(self.variables)
        for var, state in zip(variables, values):
            var_index = phi.variables.index(var)
            slice_[var_index] = state
            var_indices_to_del.append(var_index)

        index_to_keep = sorted(set(range(len(self.variables))) - set(var_indices_to_del))
        phi.variables = [phi.variables[index] for index in index_to_keep]
        phi.cardinality = [phi.cardinality[index] for index in index_to_keep]
        phi.values = phi.values[tuple(slice_)]

        if not inplace:
            return phi

    def marginalize_except(self, variables, inplace=True):
        phi = self if inplace else self.copy()
        if inplace:
            self.marginalize([var for var in phi.variables if var not in variables], inplace=True)
            self.transpose(variables)
        if not inplace:
            return self.marginalize(
                [var for var in phi.variables if var not in variables], inplace=False
            ).transpose(variables, inplace=False)

    def maximize(self, variables, inplace=True):
        phi = self if inplace else self.copy()

        for var in variables:
            if var not in phi.variables:
                raise ValueError("{var} not in scope.".format(var=var))

        var_indices = [phi.variables.index(var) for var in variables]
        index_to_keep = sorted(set(range(len(self.variables))) - set(var_indices))
        phi.variables = [phi.variables[index] for index in index_to_keep]
        phi.cardinality = [phi.cardinality[index] for index in index_to_keep]

        phi.values = np.max(phi.values, axis=tuple(var_indices))
        if not inplace:
            return phi

    def weightedsum(self, variables, weight, inplace=True):
        phi = self if inplace else self.copy()

        for var in variables:
            if var not in phi.variables:
                raise ValueError("{var} not in scope.".format(var=var))

        var_indices = [phi.variables.index(var) for var in variables]
        index_to_keep = sorted(set(range(len(self.variables))) - set(var_indices))
        phi.variables = [phi.variables[index] for index in index_to_keep]
        phi.cardinality = [phi.cardinality[index] for index in index_to_keep]

        phi.values = np.abs(phi.values)
        base = np.sqrt(np.max(phi.values)) * np.sqrt(np.min(phi.values))
        phi.values /= base

        if 1.0 / weight * np.log(np.max(phi.values)) > np.log(
            sys.float_info.max
        ) or 1.0 / weight * np.log(np.min(phi.values)) < np.log(1e-8):
            phi.values = np.max(phi.values, axis=tuple(var_indices))
        else:
            phi.values = np.power(
                np.sum(np.power(np.abs(phi.values), 1 / weight), axis=tuple(var_indices)), weight,
            )

        phi.values *= base

        if not inplace:
            return phi

    def normalize(self, variables=None, inplace=True):
        if not variables:
            variables = self.variables

        phi = self if inplace else self.copy()
        var_indices = [phi.variables.index(var) for var in variables]
        phi.values = phi.values / np.sum(phi.values, axis=tuple(var_indices), keepdims=True)
        if not inplace:
            return phi

    def product(self, phi1, name=None, inplace=True):
        phi = self if inplace else self.copy()
        if isinstance(phi1, (int, float)):
            phi.values *= phi1
            if not inplace:
                return phi
            else:
                return
        else:
            phi1 = phi1.copy()
            extra_vars = set(phi1.variables) - set(phi.variables)
            if extra_vars:
                slice_ = [slice(None)] * len(phi.variables)
                slice_.extend([np.newaxis] * len(extra_vars))

                phi.values = phi.values[slice_]
                phi.variables.extend(extra_vars)
                new_var_card = phi1.get_cardinality(extra_vars)
                phi.cardinality = np.append(
                    phi.cardinality, [new_var_card[var] for var in extra_vars]
                )

            extra_vars = set(phi.variables) - set(phi1.variables)
            if extra_vars:
                slice_ = [slice(None)] * len(phi1.variables)
                slice_.extend([np.newaxis] * len(extra_vars))

                phi1.values = phi1.values[slice_]
                phi1.variables.extend(extra_vars)

            for axis in range(phi.values.ndim):
                exchange_index = phi1.variables.index(phi.variables[axis])
                phi1.variables[axis], phi1.variables[exchange_index] = (
                    phi1.variables[exchange_index],
                    phi1.variables[axis],
                )
                phi1.values = phi1.values.swapaxes(axis, exchange_index)

            phi.values = phi.values * phi1.values

        if not inplace:
            if name:
                phi.name = name
            else:
                phi.name = str(phi.name) + str(phi1.name)
            return phi

    def __mul__(self, other):
        return self.product(other, inplace=False)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            raise TypeError("Class compared with other type of data.")

        if self.name == other.name:
            return True
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(self.name)

    def __repr__(self):
        return str(self.name)


def factor_product(*args):
    if not all((isinstance(phi, Factor) or isinstance(phi, float)) for phi in args):
        raise TypeError("Arguments must be factors or floats")

    return reduce(lambda phi1, phi2: phi1 * phi2, args).copy()
