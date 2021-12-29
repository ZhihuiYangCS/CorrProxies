import time
from itertools import combinations
from typing import List

import rootpath

from optimizer.accuracy_allocator.acc_allocator.exhuasted_seach.aa_exhausted_cost_comp import \
    AccuracyAllocatorExhaustedCC
from optimizer.accuracy_allocator.acc_allocator.exhuasted_seach.aa_exhausted_cost_func import \
    AccuracyAllocatorExhaustedCF
from optimizer.accuracy_allocator.acc_allocator.hill_climbing.aa_hill_climbing_cost_func import \
    AccuracyAllocatorHillClimbingCF

rootpath.append()
from optimizer.accuracy_allocator.acc_allocator.hill_climbing.aa_hill_climbing_cost_comp import \
    AccuracyAllocatorHillClimbingCC
from records.record import Record


class PlanGenerator:
    """
    This class generate different plans to train PPs
    """

    def __init__(self, workflow):
        self.workflow = workflow

    def generate_one_single_plan(self) -> List[List[int]]:
        """
        generate a plan for single PP
        """
        ml_filter_num = len(self.workflow.workflow) - 2
        plan = []
        sub_plan = []
        for i in range(0, ml_filter_num):
            sub_plan.append(i)
        plan.append(sub_plan)
        return plan

    def generate_each_pp_plan(self) -> List[List[int]]:
        """
        generate a plan to train PP for each ML filter
        """
        ml_filter_num = len(self.workflow.workflow) - 2
        plan = []
        for i in range(0, ml_filter_num):
            plan.append([i])
        return plan

    def generate_all_plans(self) -> List[List[List[int]]]:
        """
        generate all possible combination plans
        """
        ml_filter_num = len(self.workflow.workflow) - 2
        plans = []
        for i in range(ml_filter_num):
            combination = list(combinations(range(ml_filter_num - 1), i))
            for m in combination:
                plan = []
                start_index = 0
                for n in m:
                    plan.append(list(range(start_index, n + 1)))
                    start_index = n + 1
                sub_plan = list(range(start_index, ml_filter_num))
                if sub_plan:
                    plan.append(sub_plan)
                plans.append(plan)
        return plans

    def get_plan_ml_filters_names(self, plan: List[List[int]]) -> List[List[str]]:
        """
        get ml_filters from a plan index
        :param plan: a plan
        :return: ml_filters of this plan
        """
        plan_ml_filters = []
        ml_filters = self.workflow.get_ml_filters()
        for sub_plan in plan:
            sub_plan_ml_filters = []
            for i in sub_plan:
                sub_plan_ml_filters.append(ml_filters[i].operator_name)
            plan_ml_filters.append(sub_plan_ml_filters)
        return plan_ml_filters

    def one_plan_accuracy_allocation(self, plan: List[List[int]], workflow_samples: List[Record] = None):
        """
        perform accuracy allocation for a specified plan
        :param allocation_method:
        :param plan: a plan indicating training PPs
        :param preprocessor: a preprocessor to preprocess input records
        :param target_accuracy: target accuracy specified by a user
        :param allocation_mode: allocation mode in accuracy allocation;
                                0: resample-retrain; 1: reuse sample, retrain PPs; 2: reuse samples and reuse PPs
        :param workflow_samples: the workflow samples, if is None: doesn't reuse; else: reuse
        :return: final_pps: List[PPFilter]
                 execution_cost: float
                 final_accuracy: float
                 optimization_time: float
        """
        print("now start optimize plan = " + str(plan))
        if self.workflow.allocation_method == 0:
            accuracy_allocator = AccuracyAllocatorExhaustedCC(workflow=self.workflow, plan=plan,
                                                              workflow_samples=workflow_samples)
        elif self.workflow.allocation_method == 1:
            accuracy_allocator = AccuracyAllocatorExhaustedCF(workflow=self.workflow, plan=plan,
                                                              workflow_samples=workflow_samples)
        elif self.workflow.allocation_method == 2:
            accuracy_allocator = AccuracyAllocatorHillClimbingCC(workflow=self.workflow, plan=plan,
                                                                 workflow_samples=workflow_samples)
        else:
            accuracy_allocator = AccuracyAllocatorHillClimbingCF(workflow=self.workflow, plan=plan,
                                                                 workflow_samples=workflow_samples)
        final_pps, _, time_log = accuracy_allocator.search()
        print("now finished optimize plan = " + str(plan))
        return final_pps, time_log

    def plans_accuracy_allocation(self, workflow_samples: List[Record] = None):
        """
        perform accuracy allocation for all plans
        :param allocation_method:
        :param preprocessor: a preprocessor to preprocess input records
        :param target_accuracy: target accuracy specified by a user
        :param workflow_samples: the workflow samples
        :param allocation_mode: allocation mode in accuracy allocation;
                                0: resample-retrain; 1: reuse sample, retrain PPs; 2: reuse samples and reuse PPs
        :return: all_plans: List[List[List[int]]]
                 all_returned_pps: List[List[PPFilter]]
        """
        all_plans = [self.generate_one_single_plan(), self.generate_each_pp_plan()]
        # all_plans = self.generate_all_plans()
        # List[List[PPFilter]]
        all_returned_pps, time_logs = [], []

        for plan in all_plans:
            final_pps, time_log = self.one_plan_accuracy_allocation(plan=plan, workflow_samples=workflow_samples)
            all_returned_pps.append(final_pps)
            time_logs.append(time_log)
        return all_plans, all_returned_pps, time_logs
