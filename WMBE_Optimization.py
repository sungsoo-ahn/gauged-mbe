from Factor import Factor, factor_product
from copy import copy
import numpy as np
from time import time


class WMBE_Optimization:
    def __init__(self, model, ibound):
        self.model = model.copy()
        self.logZ = 0.0
        self.Z = 0.0
        self.baselogZ = 0.0
        self.messages = {}
        self.reparam = {var: np.zeros(model.get_nstate(var)) for var in model.variables}
        self.gauges = {
            var: Factor(
                "G" + str(var),
                [var + "_", var],
                [model.get_nstate(var), model.get_nstate(var)],
                np.eye(model.get_nstate(var)),
            )
            for var in model.variables
        }
        self.build_cover_graph(ibound)

        for var in self.variables:
            if self.parents[var]:
                self.messages[(var, self.parents[var])] = factor_product(
                    *self.get_child_factors(var),
                    *[self.messages[(child, var)] for child in self.children[var]]
                )
                self.messages[(var, self.parents[var])].marginalize([self.ovardict[var]])
                self.messages[(var, self.parents[var])].values = np.full_like(
                    self.messages[(var, self.parents[var])].values, 1.0
                )
                self.messages[(var, self.parents[var])].name = (
                    "M_" + str(var) + str(self.parents[var])
                )

        for var in reversed(self.variables):
            if self.get_child_factors(var):
                phi = factor_product(*self.get_child_factors(var))
            else:
                phi = 1.0
            if self.children[var]:
                phi = factor_product(
                    *[phi], *[self.messages[(child, var)] for child in self.children[var]]
                )
            if self.parents[var]:
                phi = factor_product(*[phi], *[self.messages[(self.parents[var], var)]])

            for child in self.children[var]:
                m = self.messages[(child, var)].copy()
                self.messages[(var, child)] = factor_product(phi, m)
                self.messages[(var, child)].marginalize(list(set(phi.variables) - set(m.variables)))
                self.messages[(var, child)].values = np.full_like(
                    self.messages[(var, child)].values, 1.0
                )
                self.messages[(var, child)].name = "M_" + str(var) + "->" + str(child)

        self.reparam_step_size = 0.1
        self.weight_step_size = 0.1
        self.gauge_step_size = 0.005

        self.elapsed_time = 0.0

    def scale_step_size(self, scale):
        self.reparam_step_size *= scale
        self.weight_step_size *= scale
        self.gauge_step_size *= scale

    def build_cover_graph(self, ibound, order=None):
        if order:
            ovariables = [self.model.variables[i] for i in order]
        else:
            ovariables = self.model.variables

        variables = []
        weights = {}
        parents = {}
        children = {}
        ovardict = {}
        nvardict = {ovar: [] for ovar in ovariables}

        working_scopes = {var: [] for var in ovariables}
        for factor in self.model.factors:
            var = next(var for var in ovariables if var in factor.variables)
            working_scopes[var].append((factor.name, set(factor.variables)))

        for (elim_idx, elim_ovar) in enumerate(ovariables):
            bucket = []
            mb_ancestor_ovars = set()
            mb_child_vars = set()
            for (working_child_var, working_ancestor_ovars) in working_scopes[elim_ovar]:
                if len(mb_ancestor_ovars.union(working_ancestor_ovars)) <= ibound + 1:
                    mb_ancestor_ovars = mb_ancestor_ovars.union(working_ancestor_ovars)
                    mb_child_vars = mb_child_vars.union({working_child_var})
                else:
                    bucket.append((mb_child_vars, mb_ancestor_ovars))
                    mb_ancestor_ovars = working_ancestor_ovars
                    mb_child_vars = {working_child_var}

            bucket.append((mb_child_vars, mb_ancestor_ovars))

            split_vars = []
            # w = np.random.random(len(bucket))
            # w /= np.sum(w)
            # print(w)
            for i, (mb_child_vars, mb_ancestor_ovars) in enumerate(bucket):
                new_var = str(elim_ovar) + "_" + str(i)
                parents[new_var] = None
                weights[new_var] = 1 / len(bucket)
                split_vars.append(new_var)

                ovardict[new_var] = elim_ovar
                nvardict[elim_ovar].append(new_var)

                children[new_var] = mb_child_vars
                for child_var in mb_child_vars:
                    parents[child_var] = new_var

                parent_ovar = next(
                    (ovar for ovar in ovariables[elim_idx + 1 :] if ovar in mb_ancestor_ovars),
                    None,
                )
                if parent_ovar:
                    working_scopes[parent_ovar].append(
                        (new_var, mb_ancestor_ovars.difference({elim_ovar}))
                    )
            variables.extend(split_vars)
            del working_scopes[elim_ovar]

        adj_factor_names = {var: [] for var in variables}
        for factor in self.model.factors:
            var = parents[factor.name]
            while var:
                if ovardict[var] in factor.variables:
                    adj_factor_names[var].append(factor.name)
                var = parents[var]

        child_factor_names = {var: [] for var in variables}
        for var in variables:
            elim_children = set()
            for child in children[var]:
                if child not in variables:
                    child_factor_names[var].append(child)
                    elim_children.add(child)
                    parents.pop(child)
            children[var] = children[var] - elim_children

        self.variables = variables
        self.parents = parents
        self.children = children  # {var : sorted(list(children[var])) for var in variables}
        self.weights = weights
        self.child_factor_names = child_factor_names
        self.ovardict = ovardict
        self.nvardict = nvardict
        self.adj_factor_names = adj_factor_names

    def normalize(self):
        for factor in self.model.factors:
            self.baselogZ += np.log(np.sum(factor.values))
            factor.normalize()

    def get_child_factors(self, var):
        return [self.model.get_factor(name) for name in self.child_factor_names[var]]

    def update_messages(self):
        for ovar in self.model.variables:
            for var in self.nvardict[ovar]:
                self.forward_pass(var)

        for ovar in self.model.variables:
            for var in self.nvardict[ovar]:
                self.backward_pass(var)

    def update_parameters(self, weight=False, gauge=False, reparam="projected_gradient"):
        t = time()
        if weight:
            for ovar in self.model.variables:
                for var in self.nvardict[ovar]:
                    self.backward_pass(var)
                self.update_weights(ovar)
                for var in self.nvardict[ovar]:
                    self.forward_pass(var)

        if gauge:
            for ovar in self.model.variables:
                for var in self.nvardict[ovar]:
                    self.backward_pass(var)
                self.update_gauges(ovar)
                for var in self.nvardict[ovar]:
                    self.forward_pass(var)
        if reparam:
            for ovar in self.model.variables:
                for var in self.nvardict[ovar]:
                    self.backward_pass(var)
                self.update_reparam(ovar)
                for var in self.nvardict[ovar]:
                    self.forward_pass(var)
        self.elapsed_time += -t + time()

    def run(self, weight=False, gauge=False, reparam=False):
        max_iter = 100
        self.update_messages()

        for t in range(max_iter):
            update_parameters(self, weight=weight, gauge=gauge, reparam=reparam)

    def forward_pass(self, var, normalize=False):
        if self.parents[var]:
            w = self.weights[var]
            self.messages[(var, self.parents[var])] = factor_product(
                *self.get_child_factors(var),
                *[self.messages[(child, var)] for child in self.children[var]]
            )
            self.messages[(var, self.parents[var])].weightedsum([self.ovardict[var]], w)
            if normalize:
                self.messages[(var, self.parents[var])].normalize()

    def backward_pass(self, var, normalize=False):
        w = self.weights[var]
        if self.get_child_factors(var):
            phi = factor_product(*self.get_child_factors(var))
        else:
            phi = 1.0
        if self.children[var]:
            phi = factor_product(
                *[phi], *[self.messages[(child, var)] for child in self.children[var]]
            )
        if self.parents[var]:
            phi = factor_product(*[phi], *[self.messages[(self.parents[var], var)]])

        phi.abs()
        for child in self.children[var]:
            wc = self.weights[child]
            m = self.messages[(child, var)].copy()
            self.messages[(var, child)] = factor_product(
                m.power(-1.0, inplace=False), phi.power(wc / w, inplace=False)
            )
            self.messages[(var, child)].weightedsum(list(set(phi.variables) - set(m.variables)), wc)
            self.messages[(var, child)].name = "M_" + str(var) + "->" + str(child)
            if normalize:
                self.messages[(var, child)].normalize()

    def apply_gauges(self, ovar):
        factor_name_list = []
        ofactors = []
        for factor in self.model.factors:
            if ovar in factor.variables:
                factor_name_list.append(factor.name)
                ofactors.append(factor)
        factor1 = self.model.get_factor(factor_name_list[0])
        org_variables = copy(factor1.variables)
        factor1.product(self.gauges[ovar])
        factor1.marginalize([ovar])
        factor1.variables[factor1.variables.index(ovar + "_")] = ovar

        factor2 = self.model.get_factor(factor_name_list[1])
        org_variables = copy(factor2.variables)
        factor2.product(self.gauges[ovar].invT(inplace=False))
        factor2.marginalize([ovar])
        factor2.variables[factor2.variables.index(ovar + "_")] = ovar

        self.gauges[ovar].values = np.eye(self.model.get_nstate(ovar))

    def compute_Z(self, add_time=False):
        if add_time:
            t = time()
        for ovar in self.model.variables:
            for var in self.nvardict[ovar]:
                self.forward_pass(var, normalize=False)

        Z = 1.0
        for var in self.variables:
            w = self.weights[var]
            if not self.parents[var]:
                phi = factor_product(
                    *self.get_child_factors(var),
                    *[self.messages[(child, var)] for child in self.children[var]]
                )
                Z *= phi.weightedsum([self.ovardict[var]], w, inplace=False).values

        self.Z = Z * np.exp(self.baselogZ)
        self.logZ = np.log(Z) + self.baselogZ
        if add_time:
            self.elapsed_time += time() - t

    def get_q(self, var):
        w = self.weights[var]
        if self.get_child_factors(var):
            phi = factor_product(*self.get_child_factors(var))
        else:
            phi = 1.0

        if self.children[var]:
            phi = factor_product(
                *[phi], *[self.messages[(child, var)] for child in self.children[var]]
            )
        if self.parents[var]:
            phi = factor_product(*[phi], *[self.messages[(self.parents[var], var)]])

        phi.abs()
        phi.power(1 / w)
        phi.normalize()
        phi.name = "q_" + str(var)
        return phi.copy()

    def update_reparam(self, ovar, method="projected_gradient"):
        if method == "projected_gradient":
            step_size = self.reparam_step_size
            b = {
                var: self.get_q(var).marginalize_except([ovar], inplace=False)
                for var in self.nvardict[ovar]
            }
            b_avg = np.sum(np.array([b[var].values for var in self.nvardict[ovar]]), 0) / len(
                self.nvardict[ovar]
            )
            for var in self.nvardict[ovar]:
                temp = b[var].copy()
                temp.values = np.exp(-step_size * (temp.values - b_avg))
                factor = self.model.get_factor(self.adj_factor_names[var][0])
                factor.product(temp)

        elif method == "fixedpoint":
            b = {
                var: self.get_q(var).marginalize_except([ovar], inplace=False)
                for var in self.nvardict[ovar]
            }
            b_avg = factor_product(
                *[b[var].power(self.weights[var], inplace=False) for var in self.nvardict[ovar]]
            )
            for var in self.nvardict[ovar]:
                temp = b[var].copy()
                temp.values = b_avg.values / b[var].values
                temp.power(self.weights[var])
                factor = self.model.get_factor(self.adj_factor_names[var][0])
                factor.product(temp)

    def update_weights(self, ovar):
        step_size = self.weight_step_size
        weights = copy(self.weights)
        H = {
            var: -np.sum(
                np.multiply(
                    self.get_q(var).values,
                    np.log(self.get_q(var).normalize([ovar], inplace=False).values),
                )
            )
            for var in self.nvardict[ovar]
        }
        H_avg = np.sum([self.weights[var] * H[var] for var in self.nvardict[ovar]])
        for var in self.nvardict[ovar]:
            weights[var] *= np.exp(-step_size * (H[var] - H_avg))

        w_sum = np.sum([self.weights[var] for var in self.nvardict[ovar]])

        for var in self.nvardict[ovar]:
            weights[var] /= w_sum

        self.weights = weights

    def update_gauges(self, ovar):
        step_size = self.gauge_step_size
        gauges_der = np.full_like(self.gauges[ovar].values, 0.0)
        for factor in self.model.get_adj_factors(ovar):
            factor_var = next(
                (var for var in self.variables if factor in self.get_child_factors(var)), None,
            )
            factor_q = self.get_q(factor_var).marginalize_except(factor.variables, inplace=False)
            factor_der = self.get_gauges_der_factor(ovar, factor)
            for gauge_idx, gauge_val in np.ndenumerate(self.gauges[ovar].values):
                gauges_der[gauge_idx] += np.sum(
                    factor_q.values / factor.values * factor_der[gauge_idx]
                )

        self.gauges[ovar].values -= step_size * gauges_der
        self.apply_gauges(ovar)

    def get_gauge_list(self, factor):
        gauge_list = {}
        for ovar in factor.variables:
            if self.model.get_adj_factors(ovar).index(factor) == 0:
                gauge_list[ovar] = self.gauges[ovar].copy()
            elif self.model.get_adj_factors(ovar).index(factor) == 1:
                gauge_list[ovar] = self.gauges[ovar].invT(inplace=False)
        return gauge_list

    def get_gauges_der_factor(self, ovariable, factor):
        factor_der = {
            gauge_idx: np.full_like(factor.values, 0.0)
            for gauge_idx, gauge_val in np.ndenumerate(self.gauges[ovariable].values)
        }
        temp = {
            gauge_idx: np.full_like(factor.values, 0.0)
            for gauge_idx, gauge_val in np.ndenumerate(self.gauges[ovariable].values)
        }
        gauge_list = {}
        if factor == self.model.get_adj_factors(ovariable)[0]:
            top_factor = factor.copy()
            factor_order = copy(factor.variables)
            for ovar in factor.variables:
                if ovar != ovariable:
                    if self.model.get_adj_factors(ovar).index(factor) == 0:
                        gauge = self.gauges[ovar].copy()
                    elif self.model.get_adj_factors(ovar).index(factor) == 1:
                        gauge = self.gauges[ovar].invT(inplace=False)

                    top_factor = factor_product(top_factor, gauge)
                    top_factor.marginalize([ovar])
                    top_factor.variables[top_factor.variables.index(ovar + "_")] = ovar

            top_factor.values = top_factor.values.transpose(
                [top_factor.variables.index(ovar) for ovar in factor_order]
            )
            top_factor.variables = copy(factor_order)

            for gauge_idx, gauge_val in np.ndenumerate(self.gauges[ovar].values):
                for factor_idx, factor_val in np.ndenumerate(factor.values):
                    if gauge_idx[0] == factor_idx[factor.variables.index(ovariable)]:
                        top_idx = list(copy(factor_idx))
                        top_idx[top_factor.variables.index(ovariable)] = gauge_idx[1]
                        factor_der[gauge_idx][factor_idx] = top_factor.values[tuple(top_idx)]

            return factor_der

        elif factor == self.model.get_adj_factors(ovariable)[1]:
            top_factor = factor.copy()
            factor_order = copy(factor.variables)

            for ovar in factor.variables:
                if ovar != ovariable:
                    if self.model.get_adj_factors(ovar).index(factor) == 0:
                        gauge = self.gauges[ovar].copy()
                    elif self.model.get_adj_factors(ovar).index(factor) == 1:
                        gauge = self.gauges[ovar].invT(inplace=False)

                    top_factor = factor_product(top_factor, gauge)
                    top_factor.marginalize([ovar])
                    top_factor.variables[top_factor.variables.index(ovar + "_")] = ovar

            top_factor.values = top_factor.values.transpose(
                [top_factor.variables.index(ovar) for ovar in factor_order]
            )
            top_factor.variables = copy(factor_order)

            for gauge_idx, gauge_val in np.ndenumerate(self.gauges[ovar].values):
                for factor_idx, factor_val in np.ndenumerate(factor.values):
                    if gauge_idx[1] == factor_idx[factor.variables.index(ovariable)]:
                        top_idx = list(copy(factor_idx))
                        top_idx[top_factor.variables.index(ovariable)] = gauge_idx[0]
                        factor_der[gauge_idx][factor_idx] -= top_factor.values[tuple(top_idx)]

            return factor_der
        else:
            print("Something wrong")
            return False


"""
"""
