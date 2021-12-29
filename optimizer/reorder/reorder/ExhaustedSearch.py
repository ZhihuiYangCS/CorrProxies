import time
from typing import List, Tuple

import rootpath

rootpath.append()
from optimizer.accuracy_allocator.plan_sample_labeler import PlanSampleLabeler
from optimizer.accuracy_allocator.pp_manager_base import PPManagerBase
from optimizer.plan_generator.workflow_labeled_sample import WorkflowLabeledSample
from records.record import Record
from optimizer.reorder.reorder_utility import print_workflow_candidates, WorkflowCandidate, reset_scan
from optimizer.reorder.reorder.reorder_base import ReorderBase


class ExhaustedSearch(ReorderBase):
    """
    exhausted search all possible order workflow, for each order workflow, allocate target-accuracy
    """

    def __init__(self, workflow):
        super().__init__(workflow=workflow)
        self.plan = self.generate_each_pp_plan()

    def generate_each_pp_plan(self) -> List[List[int]]:
        """
        generate a plan to train PP for each ML filter
        """
        ml_filter_num = len(self.workflow.workflow) - 2
        plan = []
        for i in range(0, ml_filter_num):
            plan.append([i])
        return plan

    """
    For each order, as they use the same scan operator, we need to reset the scan operator for each order.
    """

    def initialization(self):
        """
        1. generate candidate_workflows
        2. generate search tree.
        """
        time1 = time.time()
        self.generate_candidate_workflows()
        print("generate candidate workflows time = " + str(time.time() - time1))
        print_workflow_candidates(self.candidate_workflows)

    def _initialize_pp_managers(self) -> List[PPManagerBase]:
        """
        initialize pp_managers
        :return:
        """
        pp_managers = []
        for i in range(len(self.ml_filters)):
            pp_managers.append(
                PPManagerBase(workflow=self.workflow))
        return pp_managers

    def _generate_labeled_samples(self, candidate_workflow: WorkflowCandidate, plan: List[List[int]]) -> \
            Tuple[List[Record], List[List[Record]]]:
        """
        generate labeled_samples for all filters in the candidate_workflow.workflow
        """
        time1 = time.time()
        workflow_labeled_sample = WorkflowLabeledSample(workflow=candidate_workflow.workflow)
        workflow_samples = workflow_labeled_sample.generate_workflow_samples()
        time2 = time.time()
        self.label_time += (time2 - time1)
        plan_sample_labeler = PlanSampleLabeler(workflow=candidate_workflow.workflow, plan=plan,
                                                workflow_samples=workflow_samples)
        plan_sample_labeler.get_subplan_samples()
        return workflow_samples, plan_sample_labeler.subplan_samples

    def _one_workflow_allocate_accuracy(self, candidate_workflow: WorkflowCandidate):
        """
        perform accuracy allocation for the candidate_workflow
            1. reset the Scan operator
            2. initialize pp_managers
            3. generate labeled_samples for all filters in the candidate_workflow.workflow
            4. If it's necessary, update ml_operator's costs
            5. update candidate_workflow's selectivities, reductions, pp_selectivities
            6. perform accuracy allocation
        """
        reset_scan(workflow=candidate_workflow.workflow)
        pp_managers = self._initialize_pp_managers()
        plan = self.generate_each_pp_plan()
        workflow_samples, labeled_samples = self._generate_labeled_samples(candidate_workflow=candidate_workflow,
                                                                           plan=plan)
        ml_filters = candidate_workflow.workflow.get_ml_filters()
        # update constant_cost
        if self.ml_operator_cost_update:
            update_operators = self.constant_cost.check_update()
            if update_operators:
                for ml_filter in ml_filters:
                    operator_name = ml_filter.operator_name
                    if operator_name in update_operators:
                        self.constant_cost.flag = True
                        ml_filter.get_operator_cost()
                        self.constant_cost.update_operator_cost(operator_name=operator_name,
                                                                cost=ml_filter.operator_cost)
                print("\t" + str(self.constant_cost))
            else:
                self.ml_operator_cost_update = False

        # update candidate_workflow's selectivities, pp_selectivities, reductions
        for i, ml_filter in enumerate(ml_filters):
            max_pp_selectivity = min_pp_selectivity = ml_filter.get_operator_pass_rate()
            for j in range(i):
                min_pp_selectivity = min_pp_selectivity / self.workflow.target_accuracy - (
                        1 - self.workflow.target_accuracy) / self.workflow.target_accuracy
                max_pp_selectivity = max_pp_selectivity / self.workflow.target_accuracy
            candidate_workflow.min_selectivities[i] = ml_filter.get_operator_pass_rate()
            candidate_workflow.max_selectivities[i] = ml_filter.get_operator_pass_rate()
            candidate_workflow.min_pp_selectivities[i] = min_pp_selectivity
            candidate_workflow.max_pp_selectivities[i] = max_pp_selectivity
            candidate_workflow.min_reductions[i] = 1 - min_pp_selectivity * self.workflow.target_accuracy
        time1 = time.time()
        if len(self.candidate_workflows) == 1:
            final_pps, result_cost, train_time = self.accuracy_allocator.allocate_accuracy(
                labeled_samples=labeled_samples, pp_managers=pp_managers, candidate_workflow=candidate_workflow,
                first_node_input=workflow_samples)
            # self.train_time += train_time
        else:
            final_pps, result_cost, train_time = self.accuracy_allocator.allocate_accuracy(
                labeled_samples=labeled_samples, pp_managers=pp_managers, candidate_workflow=candidate_workflow,
                final_validation_flag=False, first_node_input=workflow_samples)
            # self.train_time += train_time
        time2 = time.time()
        self.aa_time += (time2 - time1)
        print("\t all candidate_workflows:")
        print_workflow_candidates(self.candidate_workflows)
        candidate_workflow.pp_filters = final_pps

    def _selected_workflow(self, candidate_workflow: WorkflowCandidate):
        """
        get pp_filters and ml_filters for the selected workflow.
        This method is used when the selected workflow has already been allocated accuracy.
        :param candidate_workflow: the selected workflow
        :return: a list of pp_filters and a list of ml_filters.
        """
        pp_filters = candidate_workflow.pp_filters
        ml_filters = candidate_workflow.workflow.get_ml_filters()
        return pp_filters, ml_filters

    def search(self):
        """
        perform accuracy allocation for each candidate_workflow, compute their costs.
        """
        time1 = time.time()
        for candidate_workflow in self.candidate_workflows:
            time11 = time.time()
            self._one_workflow_allocate_accuracy(candidate_workflow=candidate_workflow)
            print("_one_workflow_allocate_accuracy cost = " + str(time.time() - time11))
        print("\tworkflow with updated cost")
        print_workflow_candidates(workflow_candidates=self.candidate_workflows)
        self.candidate_workflow_sort()
        self.find_optimal_order_time += (time.time() - time1)
        print("\tcandidate workflows after sort")
        print_workflow_candidates(workflow_candidates=self.candidate_workflows)
        pp_filters, ml_filters = self._selected_workflow(candidate_workflow=self.candidate_workflows[0])
        time2 = time.time()
        self.reorder_time += (time2 - time1)
        self.train_time += self.accuracy_allocator.train_time
        self.aa_time = self.aa_time - self.accuracy_allocator.train_time
        time_log = str(self.label_time) + "\t" + str(self.train_time) + "\t" + str(self.aa_time) + "\t" + str(
            self.reorder_time) + "\t" + str(self.find_optimal_order_time)
        return pp_filters, ml_filters, time_log
