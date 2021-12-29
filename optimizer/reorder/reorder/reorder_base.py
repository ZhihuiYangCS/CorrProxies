import itertools
import time
from abc import ABC, abstractmethod
import rootpath

rootpath.append()
from ml_workflow.workflow_base import WorkflowBase
from optimizer.reorder.acc_alloacator_reorder_utility import AccuracyAllocatorExhaustedSearch, \
    AccuracyAllocatorHillClimbing
from optimizer.reorder.reorder_utility import ConstantCost, WorkflowCandidate, MyNode
from utility.constant import SVM_TRAIN_COST, SVM_INFER_COST, DNN_TRAIN_COST, DNN_INFER_COST, TWITTER_PP_TYPE


class ReorderBase(ABC):
    """
    a base class for reorder
        generate candidate workflows
        constant_cost initialize
        candidate_workflows sort
    """

    def __init__(self, workflow):
        """
        :param reorder_accuracy_allocator_method: reorder_accuracy_allocator_method: int
                    0: --> AccuracyAllocatorExhaustedSearch
                    1: --> AccuracyAllocatorHillClimbing
        """
        # self.workflow = workflow.get_new_workflow()
        self.workflow = workflow
        self.scan, self.sink = self.workflow.get_scan_sink()
        self.ml_filters = self.workflow.get_ml_filters()
        self.ml_filters_operator_names = self.workflow.get_ml_filters_operator_names()
        self.candidate_workflows = []
        self.constant_cost = ConstantCost(operator_names=self.ml_filters_operator_names)
        self.eta = 0.99
        self.label_time = 0
        self.train_time = 0
        self.aa_time = 0
        self.reorder_time = 0
        self.find_optimal_order_time = 0
        if self.workflow.reorder_accuracy_allocator_method == 0:
            self.accuracy_allocator = AccuracyAllocatorExhaustedSearch(workflow=self.workflow, eta=self.eta)
        else:
            self.accuracy_allocator = AccuracyAllocatorHillClimbing(workflow=self.workflow, eta=self.eta)
        # False means have not finished updating; True means have finished updating
        self.ml_operator_cost_update = True
        self._constant_cost_initialization()

    @abstractmethod
    def initialization(self):
        """
        1. generate candidate_workflows
        2. generate search tree.
        """
        pass

    @abstractmethod
    def _selected_workflow(self, candidate_workflow: WorkflowCandidate):
        """
        get pp_filters and ml_filters for the selected workflow.
        This method is used when the selected workflow has already been allocated accuracy.
        :param candidate_workflow: the selected workflow
        :return: a list of pp_filters and a list of ml_filters.
        """
        pass

    def _constant_cost_initialization(self):
        """
        initialize the constant_cost
        """
        if self.workflow.pp_type == TWITTER_PP_TYPE:
            self.constant_cost.train_cost = SVM_TRAIN_COST
            self.constant_cost.infer_cost = SVM_INFER_COST
        else:
            self.constant_cost.train_cost = DNN_TRAIN_COST
            self.constant_cost.infer_cost = DNN_INFER_COST
        for ml_filter in self.ml_filters:
            if ml_filter.operator_cost is not None:
                self.constant_cost.update_operator_cost(operator_name=ml_filter.operator_name,
                                                        cost=ml_filter.operator_cost)
                self.constant_cost.flag = True
        if not self.constant_cost.check_update():
            self.ml_operator_cost_update = False

    def workflow_check(self, ml_filters):
        """
        if two ml_filters have same ml_operators, they should be adjacent to each others.
        :param ml_filters: Tuple[MLFilter]
        :return: bool; True means this ml_filters satisfy the condition;
                       False means this ml_filters doesn't satisfy this condition
        """
        flag = True
        operator_names = []
        for index, ml_filter in enumerate(ml_filters):
            operator_name = ml_filter.ml_predicator.ml_operator.operator_name
            if operator_names and (operator_name in operator_names and operator_name != operator_names[-1]):
                flag = False
                break
            else:
                operator_names.append(operator_name)
        return flag

    def generate_candidate_workflows(self, scan_root: MyNode = None):
        """
        generate all candidate workflows
        """
        permutation_ml_filters = list(itertools.permutations(self.ml_filters))
        for ml_filter_list in permutation_ml_filters:
            if self.workflow_check(ml_filters=ml_filter_list):
                operators = [self.scan]
                for index, ml_filter in enumerate(ml_filter_list):
                    operators.append(ml_filter.get_new_object(previous_operators=ml_filter_list[0:index],
                                                              operator_name=ml_filter.operator_name.split("_")[-1]))
                operators.append(self.sink)
                candidate_workflow = WorkflowBase(operators=operators, preprocessor=self.workflow.preprocessor,
                                                  models=self.workflow.models,
                                                  workflow_name=self.workflow.workflow_name,
                                                  target_accuracy=self.workflow.target_accuracy,
                                                  scheme=self.workflow.scheme,
                                                  opt_thread_num=self.workflow.opt_thread_num,
                                                  allocation_method=self.workflow.allocation_method,
                                                  allocation_mode=self.workflow.allocation_mode,
                                                  reorder_aa_method=self.workflow.reorder_accuracy_allocator_method)
                self.candidate_workflows.append(
                    WorkflowCandidate(workflow=candidate_workflow, constant_cost=self.constant_cost, node=scan_root))

    def close(self):
        """
        After the workflow reorder, we need to close this copy workflow.
        We only need to close the scan and sink operator to close the file.
        We don't need to close ml_filters.
            Because the ml_filter's get_new_object method return a new object using previous loading model.
        """
        self.scan.close()
        self.sink.close()

    def candidate_workflow_sort(self):
        """
        sort and prune candidate workflows
        """
        for i in range(len(self.candidate_workflows)):
            if self.candidate_workflows[i] is None:
                continue
            for j in range(i + 1, len(self.candidate_workflows)):
                if self.candidate_workflows[j] is None:
                    continue
                i_min_cost = self.candidate_workflows[i].min_cost
                i_max_cost = self.candidate_workflows[i].max_cost
                j_min_cost = self.candidate_workflows[j].min_cost
                j_max_cost = self.candidate_workflows[j].max_cost

                if i_min_cost <= i_max_cost <= j_min_cost <= j_max_cost:  # no intersection
                    self.candidate_workflows[j] = None
                if j_min_cost <= j_max_cost <= i_min_cost <= i_max_cost:  # no intersection
                    self.candidate_workflows[i] = self.candidate_workflows[j]
                    self.candidate_workflows[j] = None
                if (j_min_cost <= i_min_cost <= i_max_cost <= j_max_cost) or (
                        j_min_cost <= i_min_cost <= j_max_cost <= i_max_cost):  # contain or overlap
                    workflow = self.candidate_workflows[i]
                    self.candidate_workflows[i] = self.candidate_workflows[j]
                    self.candidate_workflows[j] = workflow
        self.candidate_workflows = list(filter(lambda a: a is not None, self.candidate_workflows))

    @abstractmethod
    def search(self):
        """
        search on the generated tree.
        :return: pp_filters: a list of PPFilter
                 ml_filters: a list of MLFilter
        """
        pass

    def reorder_optimization(self):
        """
        an unified interface for reorder optimization
        :return: pp_filters: a list of PPFilter
                 ml_filters: a list of MLFilter
        """
        time1 = time.time()
        self.initialization()
        time2 = time.time()
        print("initialization time cost = " + str(time2 - time1))
        pp_filters, ml_filters, time_log = self.search()
        # self.reorder_close()
        time3 = time.time()
        print("search time cost = " + str(time3 - time2))
        return pp_filters, ml_filters, time_log
