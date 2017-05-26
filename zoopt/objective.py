
from zoopt.solution import Solution
import numpy as np
from zoopt.utils.zoo_global import pos_inf

"""
The class Objective represents the objective function and its associated variables

Author:
    Yuren Liu
"""


class Objective:
    def __init__(self, func=None, dim=None, constraint=None):
        # Objective function defined by the user
        self.__func = func
        # Number of dimensions, dimension bounds are in the dim object
        self.__dim = dim
        # the function for inheriting solution attachment
        self.__inherit = self.default_inherit
        # the constraint function
        self.__constraint = constraint
        # the history of optimization
        self.__history = []
        # the stable instances during the policy search
        self.__stable_ins = []

    # Construct a solution from x
    def construct_solution(self, x, parent=None):
        new_solution = Solution()
        new_solution.set_x(x)
        new_solution.set_attach(self.__inherit(parent))
        # new_solution.set_value(self.__func(new_solution)) # evaluation should be invoked explicitly
        return new_solution

    # evaluate the objective function of a solution
    def eval(self, solution):
        solution.set_value(self.__func(solution))
        self.__history.append(solution.get_value())

    def accurate_eval(self, solution):
        turns = 100
        solution_values = []
        for i in range(turns):
            solution_values.append(self.__func(solution))
        solution.set_value(np.mean(solution_values))
        solution.set_std(np.std(solution_values))
        self.__stable_ins.append(solution)

    def eval_constraint(self, solution):
        solution.set_value( [self.__func(solution), self.__constraint(solution)])
        self.__history.append(solution.get_value())

    # set the optimization function
    def set_func(self, func):
        self.__func = func

    # get the optimization function
    def get_func(self):
        return self.__func

    # set the dimension object
    def set_dim(self, dim):
        self.__dim = dim

    # get the dimension object
    def get_dim(self):
        return self.__dim

    # set the attachment inheritance function
    def set_inherit_func(self, inherit_func):
        self.__inherit=inherit_func

    # get the attachment inheritance function
    def get_inherit_func(self):
        return self.__inherit

    # set the constraint function
    def set_constraint(self, constraint):
        self.__constraint = constraint
        return

    # return the constraint function
    def get_constraint(self):
        return self.__constraint

    # get the optimization history
    def get_history(self):
        return self.__history

    # get best stable ins
    def get_best_stable_ins(self):
        if len(self.__stable_ins) == 0:
            return None
        best_ins = self.__stable_ins[0]
        for ins in self.__stable_ins:
            if ins.get_value() < best_ins.get_value():
                best_ins = ins
        return best_ins

    # get the best-so-far history
    def get_history_bestsofar(self):
        history_bestsofar = []
        bestsofar = pos_inf
        for i in range(len(self.__history)):
            if self.__history[i] < bestsofar:
                bestsofar = self.__history[i]
            history_bestsofar.append(bestsofar)
        return history_bestsofar

    # clean the optimization history
    def clean_history(self):
        self.__history=[]

    @staticmethod
    def default_inherit(parent=None):
        return None
