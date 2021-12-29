import time
from typing import List, Union

import rootpath

rootpath.append()
from optimizer.accuracy_allocator.acc_allocator.accuracy_allocator_cost_func_base import AccuracyAllocatorCostFuncBase
from optimizer.accuracy_allocator.acc_allocator.accuracy_allocator_utility import AccCombPPFiltersList, \
    AccCombPPFiltersMap
from records.record import Record


class AccuracyAllocatorHillClimbingCF(AccuracyAllocatorCostFuncBase):
    """
    this method only works when plans are singlePP plan and multiplePPs plan.
    For example, the workflow is A(OP_1) -> B(OP_1) -> C(OP_2)
        singlePP plan is [[A,B,C]] and multiplePPs plan is [[A], [B], [C]]
        this method doesn't work for plan [[A],[B,C]]
    allocation_mode: 0 ---- resample retrain
                     1 ---- reuse sample, retrain
                     2 ---- reuse sample, reuse PPs
    """

    def __init__(self, workflow, plan: List[List[int]], workflow_samples: List[Record] = None):
        super().__init__(workflow=workflow, plan=plan, workflow_samples=workflow_samples)

    def _one_step_climbing(self, i: int, j: int, accuracy_combination: List[float],
                           search_acc_comb_pp_list: AccCombPPFiltersList) -> Union[AccCombPPFiltersMap, None]:
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

    def _allocate_accuracy_one_hill_climbing(self, accuracy_combination: List[float],
                                             search_acc_comb_pp_list: AccCombPPFiltersList):
        acc_pps_list = AccCombPPFiltersList()
        for i in range(len(accuracy_combination)):
            for j in range(i + 1, len(accuracy_combination)):
                print(str(i) + "," + str(j))
                acc_comb_pp_1 = self._one_step_climbing(i=i, j=j, accuracy_combination=accuracy_combination,
                                                        search_acc_comb_pp_list=search_acc_comb_pp_list)
                if acc_comb_pp_1 is not None:
                    acc_pps_list.add(acc_pp=acc_comb_pp_1)
                acc_comb_pp_2 = self._one_step_climbing(i=j, j=i, accuracy_combination=accuracy_combination,
                                                        search_acc_comb_pp_list=search_acc_comb_pp_list)
                if acc_comb_pp_2 is not None:
                    acc_pps_list.add(acc_pp=acc_comb_pp_2)
        return acc_pps_list.min()

    def _allocate_accuracy_hill_climbing(self, accuracy_combination: List[float],
                                         search_acc_comb_pp_list: AccCombPPFiltersList):
        old_acc_comb_pps = search_acc_comb_pp_list.search(accuracy_combination=accuracy_combination)
        if old_acc_comb_pps is None:
            old_acc_comb_pps = self.one_accuracy_combination_cost(accuracy_combination=accuracy_combination)
            search_acc_comb_pp_list.add(acc_pp=old_acc_comb_pps)
        new_acc_comb_pps = self._allocate_accuracy_one_hill_climbing(accuracy_combination=accuracy_combination,
                                                                     search_acc_comb_pp_list=search_acc_comb_pp_list)
        count = 0
        print("the " + str(count) + " step, old " + str(old_acc_comb_pps) + "\nnew = " + str(new_acc_comb_pps))
        while new_acc_comb_pps is not None and new_acc_comb_pps.cost < old_acc_comb_pps.cost * 1.01:
            old_acc_comb_pps = new_acc_comb_pps
            new_acc_comb_pps = self._allocate_accuracy_one_hill_climbing(
                accuracy_combination=old_acc_comb_pps.accuracy_combination,
                search_acc_comb_pp_list=search_acc_comb_pp_list)
            count += 1
            print("the " + str(count) + " step, old " + str(old_acc_comb_pps) + "\nnew = " + str(new_acc_comb_pps))

    def search(self):
        """
        finding the minimal cost accuracy combination, first give a arbitrary max value
        :return: a list of PPFilter trained on the search accuracy combination which has a minial cost.
        """
        time1 = time.time()
        search_acc_comb_pp_list = AccCombPPFiltersList()
        initial_accuracy = [self.workflow.target_accuracy] + [1.00] * (len(self.plan) - 1)
        self.initialize_plan_pp_managers()
        self._allocate_accuracy_hill_climbing(accuracy_combination=initial_accuracy,
                                              search_acc_comb_pp_list=search_acc_comb_pp_list)
        # one jump, search from another start point
        if len(initial_accuracy) >= 2:
            initial_accuracy = [1.00] * (len(self.plan) - 1) + [self.workflow.target_accuracy]
            self._allocate_accuracy_hill_climbing(accuracy_combination=initial_accuracy,
                                                  search_acc_comb_pp_list=search_acc_comb_pp_list)
        result_acc_comb_pps = search_acc_comb_pp_list.min()
        print("final accuracy_combination = " + str(result_acc_comb_pps.accuracy_combination) + "final cost = " + str(
            result_acc_comb_pps.cost) + "\tfrom retrain or not: " + str(result_acc_comb_pps.retrain_flag))
        if result_acc_comb_pps.retrain_flag:
            print("no need to retrain")
            final_pps = result_acc_comb_pps.pp_filters
            result_cost = result_acc_comb_pps.cost
        else:
            print("now retrain")
            final_pps, result_cost = self.retrain_final_pps(
                result_accuracy_combination=result_acc_comb_pps.accuracy_combination,
                first_pp=result_acc_comb_pps.pp_filters[0])
        time2 = time.time()
        self.aa_time += (time2 - time1)
        time_log = str(self.label_time) + "\t" + str(self.train_time) + "\t" + str(self.aa_time) + "\t" + str(
            self.reorder_time)
        return final_pps, result_cost, time_log
