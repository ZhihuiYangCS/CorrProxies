import sys
import time
from typing import List

import rootpath

rootpath.append()
from optimizer.accuracy_allocator.acc_allocator.accuracy_allocator_cost_func_base import AccuracyAllocatorCostFuncBase
from optimizer.accuracy_allocator.acc_allocator.accuracy_allocator_utility import generate_exhausted_search_space, \
    AccCombPPFiltersMap
from records.record import Record


class AccuracyAllocatorExhaustedCF(AccuracyAllocatorCostFuncBase):
    """
    exhausted search
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

    def search(self):
        """
        an abstract method, implement exhausted search and provide an outer interface to perform accuracy allocation
        finding the minimal cost accuracy combination, first give a arbitrary max value
        :return: a list of PPFilter trained on the search accuracy combination which has a minial cost.
        """
        time0 = time.time()
        result_acc_comb_pps = AccCombPPFiltersMap(accuracy_combination=[], cost=sys.maxsize, pp_filters=[],
                                                  retrain_flag=True)
        self.search_space = generate_exhausted_search_space(accuracy_num=len(self.plan),
                                                            target_accuracy=self.workflow.target_accuracy, eta=self.eta)
        self.initialize_plan_pp_managers()
        temp_index = 0
        for accuracy_list in self.search_space:
            print("the " + str(temp_index) + "th search step, accuracy = " + str(accuracy_list))
            time1 = time.time()
            acc_comb_pps = self.one_accuracy_combination_cost(accuracy_combination=accuracy_list)
            if acc_comb_pps.cost < result_acc_comb_pps.cost:
                result_acc_comb_pps = acc_comb_pps
            print("\tone iteration time = " + str(time.time() - time1))
            temp_index += 1
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
        self.aa_time += (time2 - time0)
        time_log = str(self.label_time) + "\t" + str(self.train_time) + "\t" + str(self.aa_time) + "\t" + str(
            self.reorder_time)
        return final_pps, result_cost, time_log
