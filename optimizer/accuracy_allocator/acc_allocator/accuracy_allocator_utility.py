import sys
from itertools import product
from typing import List, Union
import rootpath

rootpath.append()
from optimizer.PPs.pp_filter import PPFilter
from optimizer.reorder.reorder_utility import WorkflowCandidate


def accuracy_combination_round(accuracy_combination: List[float]) -> List[float]:
    """
    round of accuracy_combination
    """
    result_acc_comb = [round(accuracy, 2) for accuracy in accuracy_combination]
    return result_acc_comb


def generate_exhausted_search_space(accuracy_num: int, target_accuracy: float, eta: float) -> List[List[float]]:
    """
    generate exhausted search space.
    :param accuracy_num: the accuracy_num specifies the accuracy_combination dimension.
                         It also is the dimension to perform accuracy allocation
    :param target_accuracy: target_accuracy
    :param eta: the step
    :return: a exhuasted search space. List[List[float]]
    """
    search_space = []
    one_dim_discrete = []
    accuracy = target_accuracy
    while 1.0 >= accuracy >= target_accuracy:
        one_dim_discrete.append(accuracy)
        accuracy /= eta
    one_dim_discrete.append(1.0)
    # print("one dimension discrete = " + str(one_dim_discrete))
    iter_loop_discrete = []
    for i in range(accuracy_num):
        one_dim_discrete_copy = one_dim_discrete.copy()
        iter_loop_discrete.append(one_dim_discrete_copy)
    for one_accuracy_combination in product(*iter_loop_discrete):
        combination = []
        multiply = 1
        for accuracy in one_accuracy_combination:
            combination.append(accuracy)
            multiply *= accuracy
        if target_accuracy <= multiply <= target_accuracy + 0.005:
            search_space.append(combination)
    return search_space


class AccCombPPFiltersMap:
    """
    a data structure for searched accuracy_combination
    """

    def __init__(self, accuracy_combination: List[float], cost: float, pp_filters: List[PPFilter],
                 retrain_flag: bool, candidate_workflow: Union[WorkflowCandidate, None] = None):
        """

        :param accuracy_combination: the search accuracy_combination
        :param cost: the cost of the candidate_workflow under the accuracy_combination
        :param pp_filters: a list of PPFilters obtained under the accuracy_combination
        :param retrain_flag: a flag indicating the list of PPFilters are from retrain or not.
                             True: the list of PPFilters are from retrain
                             False: the list of PPFilters are not from retrain
        :param candidate_workflow: a candidate_workflow updated under the accuracy_combination
        """
        self.accuracy_combination = accuracy_combination
        self.accuracy_combination_round = accuracy_combination_round(accuracy_combination=accuracy_combination)
        self.cost = cost
        self.pp_filters = pp_filters
        self.retrain_flag = retrain_flag
        self.candidate_workflow = candidate_workflow

    def __str__(self):
        return "accuracy_combination = " + str(self.accuracy_combination) + "\t cost = " + str(
            self.cost) + "\tretrain_flag = " + str(self.retrain_flag)

    __repr__ = __str__


class AccCombPPFiltersList:
    """
    a list of AccCombPPFiltersMap,
        support adding an AccCombPPFiltersMap object,
        and search for an AccCombPPFiltersMap.
    """

    def __init__(self):
        self.acc_pp_list: List[AccCombPPFiltersMap] = []

    def add(self, acc_pp: AccCombPPFiltersMap):
        """
        add an AccCombPPFiltersMap object
        """
        self.acc_pp_list.append(acc_pp)

    def search(self, accuracy_combination: List[float]) -> Union[AccCombPPFiltersMap, None]:
        """
        search for an AccCombPPFiltersMap based on accuracy_combination
        """
        accuracy_combination_round_value = accuracy_combination_round(accuracy_combination=accuracy_combination)
        for acc_pp in self.acc_pp_list:
            if acc_pp.accuracy_combination_round == accuracy_combination_round_value:
                return acc_pp
        return None

    def min(self):
        """
        return an AccCombPPFiltersMap object with the minimal cost
        """
        min_cost = sys.maxsize
        result_acc_comb_pp = None
        for i in range(len(self.acc_pp_list) - 1, -1, -1):
            if self.acc_pp_list[i].cost < min_cost:
                result_acc_comb_pp = self.acc_pp_list[i]
                min_cost = self.acc_pp_list[i].cost
        return result_acc_comb_pp

    def __str__(self):
        info = '\n '.join(map(str, self.acc_pp_list))
        return f"AccCombPPFiltersList:\n[{info}]"

    __repr__ = __str__
