from typing import Any, List, Union

import rootpath

rootpath.append()
from optimizer.accuracy_allocator.acc_allocator.accuracy_allocator_utility import AccCombPPFiltersMap, \
    AccCombPPFiltersList
from ml_workflow.workflow_base import WorkflowBase
from optimizer.compare_methods.microsoft_pp_base import MicrosoftPPBase


class MicrosoftPPDP(MicrosoftPPBase):
    """
    hill_climbing search
    implementing the method in the paper "Accelerating Machine Learning Inference with Probabilistic Predicates"
    """

    def __init__(self, workflow: WorkflowBase):
        super().__init__(workflow=workflow)

    def _one_step_dynamic_programming(self, i: int, j: int, accuracy_combination: List[float],
                                      search_acc_comb_pp_list: AccCombPPFiltersList) -> Union[
        AccCombPPFiltersMap, None]:
        accuracy_combination_copy = accuracy_combination.copy()
        accuracy_combination_copy[i] *= self.eta
        accuracy_combination_copy[j] /= self.eta
        if self.workflow.target_accuracy <= accuracy_combination_copy[i] <= 1.0 and \
                self.workflow.target_accuracy <= accuracy_combination_copy[j] <= 1.0:
            acc_comb_pp = search_acc_comb_pp_list.search(accuracy_combination=accuracy_combination_copy)
            if acc_comb_pp is None:
                acc_comb_pp = self.one_accuracy_combination_cost(accuracy_combination=accuracy_combination_copy)
                search_acc_comb_pp_list.add(acc_pp=acc_comb_pp)
                return acc_comb_pp
        return None

    def _allocate_accuracy_one_dynamic_programming(self, accuracy_combination: List[float],
                                                   search_acc_comb_pp_list: AccCombPPFiltersList):
        acc_pps_list = AccCombPPFiltersList()
        for i in range(len(accuracy_combination)):
            for j in range(i + 1, len(accuracy_combination)):
                print(str(i) + "," + str(j))
                acc_comb_pp_1 = self._one_step_dynamic_programming(i=i, j=j, accuracy_combination=accuracy_combination,
                                                                   search_acc_comb_pp_list=search_acc_comb_pp_list)
                if acc_comb_pp_1 is not None:
                    acc_pps_list.add(acc_pp=acc_comb_pp_1)
                acc_comb_pp_2 = self._one_step_dynamic_programming(i=j, j=i, accuracy_combination=accuracy_combination,
                                                                   search_acc_comb_pp_list=search_acc_comb_pp_list)
                if acc_comb_pp_2 is not None:
                    acc_pps_list.add(acc_pp=acc_comb_pp_2)
        return acc_pps_list.min()

    def _allocate_accuracy_dynamic_programming(self, accuracy_combination: List[float],
                                               search_acc_comb_pp_list: AccCombPPFiltersList):
        old_acc_comb_pps = search_acc_comb_pp_list.search(accuracy_combination=accuracy_combination)
        if old_acc_comb_pps is None:
            old_acc_comb_pps = self.one_accuracy_combination_cost(accuracy_combination=accuracy_combination)
            search_acc_comb_pp_list.add(acc_pp=old_acc_comb_pps)
        new_acc_comb_pps = self._allocate_accuracy_one_dynamic_programming(accuracy_combination=accuracy_combination,
                                                                           search_acc_comb_pp_list=search_acc_comb_pp_list)
        count = 0
        print("the " + str(count) + " step, old " + str(old_acc_comb_pps) + "\nnew = " + str(new_acc_comb_pps))
        while new_acc_comb_pps is not None and new_acc_comb_pps.cost < old_acc_comb_pps.cost * 1.001:
            old_acc_comb_pps = new_acc_comb_pps
            new_acc_comb_pps = self._allocate_accuracy_one_dynamic_programming(
                accuracy_combination=old_acc_comb_pps.accuracy_combination,
                search_acc_comb_pp_list=search_acc_comb_pp_list)
            count += 1
            print("the " + str(count) + " step, old " + str(old_acc_comb_pps) + "\nnew = " + str(new_acc_comb_pps))

    def search(self):
        """
        exhausted search
        :return:  -> List[PPFilter]
        """
        self.ml_operator_costs = self.workflow.get_ml_operator_costs()
        print("ml_operator_costs = " + str(self.ml_operator_costs))
        search_acc_comb_pp_list = AccCombPPFiltersList()
        initial_accuracy = [self.workflow.target_accuracy] + [1.00] * (len(self.ml_filters) - 1)
        self._allocate_accuracy_dynamic_programming(accuracy_combination=initial_accuracy,
                                                    search_acc_comb_pp_list=search_acc_comb_pp_list)

        result_acc_comb_pps = search_acc_comb_pp_list.min()
        print("final accuracy_combination = " + str(result_acc_comb_pps.accuracy_combination) + "final cost = " + str(
            result_acc_comb_pps.cost))
        print(" selected accuracy_reduction allocation = ")
        for i in range(len(result_acc_comb_pps.pp_filters)):
            print(result_acc_comb_pps.pp_filters[i].accuracy_info_dict)
        return result_acc_comb_pps.pp_filters

