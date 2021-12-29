import time
import rootpath

rootpath.append()
from optimizer.plan_generator.workflow_labeled_sample import WorkflowLabeledSample
from optimizer.reorder.reorder.reorder_base import ReorderBase
from optimizer.reorder.reorder_utility import WorkflowCandidate, print_workflow_candidates, reset_scan


class ReorderBaseline(ReorderBase):
    def __init__(self, workflow):
        super().__init__(workflow=workflow)

    def initialization(self):
        """
        1. generate candidate_workflows
        2. generate search tree.
        """
        time1 = time.time()
        self.generate_candidate_workflows()
        print("generate candidate workflows time = " + str(time.time() - time1))
        print_workflow_candidates(self.candidate_workflows)

    def _selected_workflow(self, candidate_workflow: WorkflowCandidate):
        ml_filters = candidate_workflow.workflow.get_ml_filters()
        return [], ml_filters

    def workflow_cost(self, candidate_workflow: WorkflowCandidate):
        """
        compute the candidate_workflow's cost
        """
        reset_scan(workflow=candidate_workflow.workflow)
        time1 = time.time()
        workflow_labeled_sample = WorkflowLabeledSample(workflow=candidate_workflow.workflow)
        workflow_samples = workflow_labeled_sample.generate_workflow_samples()
        time2 = time.time()
        self.label_time += (time2 - time1)
        print("generate_workflow_samples cost = " + str(time2 - time1))

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
        cost, selectivity = 0, 1
        for ml_filter in ml_filters:
            # operator_name = ml_filter.ml_predicator.ml_operator.operator_name
            cost += selectivity * self.constant_cost.min_costs[
                self.constant_cost.operator_names.index(ml_filter.operator_name)]
            ml_filter.get_operator_pass_rate()
            selectivity *= ml_filter.operator_cost
        candidate_workflow.min_cost = cost
        candidate_workflow.max_cost = cost

    def search(self):
        """
        compute each candidate_workflow's cost and select the minimal cost workflow
        """
        time1 = time.time()
        for candidate_workflow in self.candidate_workflows:
            time10 = time.time()
            self.workflow_cost(candidate_workflow=candidate_workflow)
            print("compute one workflow cost = " + str(time.time() - time10))
        print("\tworkflow with updated cost")
        print_workflow_candidates(workflow_candidates=self.candidate_workflows)
        self.candidate_workflow_sort()
        print("\tcandidate workflows after sort")
        print_workflow_candidates(workflow_candidates=self.candidate_workflows)
        pp_filters, ml_filters = self._selected_workflow(candidate_workflow=self.candidate_workflows[0])
        time2 = time.time()
        self.reorder_time += (time2 - time1)
        time_log = str(self.label_time) + "\t" + str(self.train_time) + "\t" + str(self.aa_time) + "\t" + str(
            self.reorder_time)
        return pp_filters, ml_filters, time_log
