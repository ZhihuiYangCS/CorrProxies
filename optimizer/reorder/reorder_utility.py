import sys
from typing import List
from anytree import NodeMixin, RenderTree
import rootpath

rootpath.append()
from operators.operator_base.operator_parallel import OperatorParallel
from operators.ml_filter import MLFilter
from optimizer.PPs.pp_filter import PPFilter
from records.record import Record
from utility.constant import SVM_TRAIN_COST, SVM_INFER_COST, TWITTER_TYPE, UCF101_TYPE

"""
utility classes
"""


class MyNode(NodeMixin):
    """
    define our node class
    """

    def __init__(self, node_id: int, node_type: str, operator: OperatorParallel, parent=None, children=None):
        super(MyNode, self).__init__()
        self.node_id = node_id
        self.node_type = node_type
        self.operator = None
        self.operator = operator
        if parent is not None:
            self.parent = parent
        if children is not None:
            self.children = children
        self.cache = []
        self.labeled_samples = []  # s_node
        self.process_num = 0  # s_node
        self.pass_num = 0  # s_node
        self.pp_manager = None  # t_node
        self.max_pp_filter = None  # t_node
        self.min_pp_filter = None  # t_node
        self.first_s_node_input = []  # first s_node input
        self.name = str(self.node_id) + "-" + self.node_type + "-" + self.operator.operator_name + "-" + str(
            len(self.cache)) + "-" + str(len(self.labeled_samples)) + "-" + str(len(self.first_s_node_input))

    def get_all_pass_nodes(self):
        """
        get all pass nodes for the current node
        :return: nodes:List[MyNode]
        """
        nodes = list(self.path)
        return nodes

    def get_pass_st_nodes(self):
        """
        get all pass s_nodes and t_nodes, respectively.
        :return: s_nodes:List[MyNode], t_nodes:List[MyNode]
        """
        s_nodes = []
        t_nodes = []
        nodes = list(self.path)
        nodes.pop()  # remove the last element, self
        for node in nodes:
            if node.node_type == "S":
                s_nodes.append(node)
            if node.node_type == "T":
                t_nodes.append(node)
        return s_nodes, t_nodes

    def get_pass_workflow_nodes(self):
        """
        get this nodes pass workflow nodes, including itself
        :return: a list of MyNode
        """
        result_nodes = []
        s_nodes, t_nodes = self.get_pass_st_nodes()
        if self.node_type == "S":
            s_nodes.append(self)
            result_nodes = s_nodes
        if self.node_type == "T":
            t_nodes.append(self)
            result_nodes = t_nodes
        return result_nodes

    def __str__(self):
        return str(self.node_id) + "-" + self.node_type + "-" + self.operator.operator_name + "-" + str(
            len(self.cache)) + "-" + str(len(self.labeled_samples)) + "-" + str(len(self.first_s_node_input))

    __repr__ = __str__


class ConstantCost:
    """
    Update this ConstantCost from ml_operators to ml_filters
    This class encapsulate all constant costs including ml_filters costs, train_cost and infer_cost
    """

    def __init__(self, operator_names: List[str]):
        self.operator_names = operator_names
        self.operator_num = len(self.operator_names)
        self.min_costs = [0] * self.operator_num
        self.max_costs = [sys.maxsize] * self.operator_num
        self.train_cost = SVM_TRAIN_COST
        self.infer_cost = SVM_INFER_COST
        # if operator_costs are updated, self.flag = True; else: self.flag = False
        self.flag = False

    def check_update(self) -> List[str]:
        """
        check which ml_operators are not updated
        :return: a list of not updated operator names
        """
        not_equals = []
        for i in range(self.operator_num):
            if self.min_costs[i] != self.max_costs[i]:
                not_equals.append(self.operator_names[i])
        return not_equals

    def update_operator_cost(self, operator_name: str, cost: float):
        """
        update the cost of the operator specified using operator_name
        :param operator_name: the name of the operator
        :param cost: the cost of the ml_operator
        """
        index: int = self.operator_names.index(operator_name)
        if self.min_costs[index] != self.max_costs[index]:
            self.min_costs[index] = cost
            self.max_costs[index] = cost

    def update_infer_cost(self, cost: float):
        """
        update the infer cost
        """
        if self.infer_cost == 0:
            self.infer_cost = cost

    def __str__(self):
        return "operator_names = " + str(self.operator_names) + "\tmax_costs = " + str(self.max_costs) + \
               "\tmin_costs = " + str(self.min_costs) + "\tinfer_cost = " + str(self.infer_cost)

    __repr__ = __str__


class WorkflowCandidate:
    """
    a candidate workflow including its cost computation
    """

    def __init__(self, workflow, constant_cost: ConstantCost, node: MyNode = None):
        self.workflow = workflow
        self.ml_filters = self.workflow.get_ml_filters()
        self.ml_filters_num = len(self.ml_filters)
        self.constant_cost = constant_cost
        self.min_selectivities = [0] * self.ml_filters_num
        self.max_selectivities = [1] * self.ml_filters_num
        self.min_reductions = [1] * self.ml_filters_num
        self.max_reductions = [0] * self.ml_filters_num
        self.min_pp_selectivities = [0] * self.ml_filters_num
        self.max_pp_selectivities = [1] * self.ml_filters_num
        self.max_accuracy = [1] * self.ml_filters_num
        self.min_accuracy = [self.workflow.target_accuracy] * self.ml_filters_num
        self.min_cost = self.get_min_cost()
        self.max_cost = self.get_max_cost()
        # this attribute self.pp_filters works only in the exhausted_search method
        self.pp_filters = None
        self.node = node
        self.flag = False

    def candidate_workflow_copy(self):
        """
        copy candidate_workflow, and return self's copy
        :return: a candidate_workflow
        """
        candidate_workflow = WorkflowCandidate(workflow=self.workflow, constant_cost=self.constant_cost, node=self.node)
        candidate_workflow.min_selectivities = self.min_selectivities.copy()
        candidate_workflow.max_selectivities = self.max_selectivities.copy()
        candidate_workflow.min_reductions = self.min_reductions.copy()
        candidate_workflow.max_reductions = self.max_reductions.copy()
        candidate_workflow.min_pp_selectivities = self.min_pp_selectivities.copy()
        candidate_workflow.max_pp_selectivities = self.max_pp_selectivities.copy()
        candidate_workflow.min_accuracy = self.min_accuracy.copy()
        candidate_workflow.max_accuracy = self.max_accuracy.copy()
        candidate_workflow.min_cost = self.min_cost
        candidate_workflow.max_cost = self.max_cost
        return candidate_workflow

    def candidate_workflow_reset(self, candidate_workflow):
        """
        using param candidate_workflow to reset self.
        """
        self.candidate_workflow_clear()
        self.min_selectivities = candidate_workflow.min_selectivities.copy()
        self.max_selectivities = candidate_workflow.max_selectivities.copy()
        self.min_reductions = candidate_workflow.min_reductions.copy()
        self.max_reductions = candidate_workflow.max_reductions.copy()
        self.min_pp_selectivities = candidate_workflow.min_pp_selectivities.copy()
        self.max_pp_selectivities = candidate_workflow.max_pp_selectivities.copy()
        self.min_accuracy = candidate_workflow.min_accuracy.copy()
        self.max_accuracy = candidate_workflow.max_accuracy.copy()
        self.min_cost = candidate_workflow.min_cost
        self.max_cost = candidate_workflow.max_cost
        # self.flag = True

    def candidate_workflow_clear(self):
        """
        clear the candidate_workflow
        """
        self.min_selectivities.clear()
        self.max_selectivities.clear()
        self.min_reductions.clear()
        self.max_reductions.clear()
        self.min_pp_selectivities.clear()
        self.max_pp_selectivities.clear()
        self.min_accuracy.clear()
        self.max_accuracy.clear()

    def get_min_cost(self) -> float:
        """
        compute the minimal cost of the workflow
        :return: the minimal cost
        """
        operator_name = None
        cost, selectivity = 0, 1
        save_time, pass_rate, last_unit_pass_rate = 0, 1, 1
        for i in range(self.ml_filters_num):
            if operator_name is None or (operator_name is not None and operator_name !=
                                         self.ml_filters[i].ml_predicator.ml_operator.operator_name):
                cost += self.constant_cost.min_costs[self.constant_cost.operator_names.index(
                    self.ml_filters[i].operator_name)] * selectivity
                pass_rate = last_unit_pass_rate
            operator_name = self.ml_filters[i].ml_predicator.ml_operator.operator_name
            selectivity *= self.min_selectivities[i]
            if i == 0:
                save_time += pass_rate * (self.min_reductions[i] * self.constant_cost.min_costs[
                    self.constant_cost.operator_names.index(self.ml_filters[i].operator_name)])
            else:
                save_time += pass_rate * (self.min_reductions[i] * self.constant_cost.min_costs[
                    self.constant_cost.operator_names.index(
                        self.ml_filters[i].operator_name)] - self.constant_cost.infer_cost)
            last_unit_pass_rate *= pass_rate * self.min_pp_selectivities[i] * self.min_accuracy[i]
            pass_rate *= (1 - self.min_reductions[i])
            # print("cost = " + str(cost) + "\tsave_time = " + str(save_time))
        return cost - save_time

    def get_max_cost(self) -> float:
        """
        compute the maximal cost of the workflow
        :return: the maximal cost
        """
        operator_name = None
        cost, selectivity = 0, 1
        save_time, pass_rate, last_unit_pass_rate = 0, 1, 1
        for i in range(self.ml_filters_num):
            if operator_name is None or (operator_name is not None and operator_name !=
                                         self.ml_filters[i].ml_predicator.ml_operator.operator_name):
                cost += self.constant_cost.max_costs[self.constant_cost.operator_names.index(
                    self.ml_filters[i].operator_name)] * selectivity
                pass_rate = last_unit_pass_rate
            operator_name = self.ml_filters[i].ml_predicator.ml_operator.operator_name
            selectivity *= self.max_selectivities[i]
            if i == 0:
                save_time += pass_rate * (self.max_reductions[i] * self.constant_cost.max_costs[
                    self.constant_cost.operator_names.index(self.ml_filters[i].operator_name)])
            else:
                save_time += pass_rate * (self.max_reductions[i] * self.constant_cost.max_costs[
                    self.constant_cost.operator_names.index(
                        self.ml_filters[i].operator_name)] - self.constant_cost.infer_cost)
            last_unit_pass_rate *= pass_rate * self.max_pp_selectivities[i] * self.max_accuracy[i]
            pass_rate *= (1 - self.max_reductions[i])
            # print("cost = " + str(cost) + "\tsave_time = " + str(save_time))
        return cost - save_time

    def get_cost_cc(self, samples: List[Record], pp_filters: List[PPFilter], ml_filters: List[MLFilter]) -> float:
        """
        compute a workflow's cost cost computation method
        """
        cost = 0
        all_samples = samples.copy()
        all_samples_num = len(all_samples)
        for pp_filter in pp_filters:
            cost += len(all_samples) * pp_filter.accuracy_info_dict.infer_cost
            if self.workflow.workflow_type == UCF101_TYPE and self.workflow.opt_thread_num == 0:
                pp_batch_output = pp_filter.process_batch(batch=all_samples, model=self.workflow.models[
                    self.workflow.preprocessor.operator_name])
            else:
                pp_batch_output = pp_filter.multiple_threads_batch(batch=all_samples,
                                                                   num_process=self.workflow.opt_thread_num)
            all_samples = pp_batch_output.returned_batch
        for ml_filter in ml_filters:
            cost += len(all_samples) * self.constant_cost.max_costs[
                self.constant_cost.operator_names.index(ml_filter.operator_name)]
            if self.workflow.workflow_type == UCF101_TYPE and self.workflow.opt_thread_num == 0:
                filter_batch_output = ml_filter.process_batch(batch=all_samples, model=self.workflow.models[
                    ml_filter.ml_predicator.ml_operator.operator_name])
            else:
                filter_batch_output = ml_filter.multiple_threads_batch(batch=all_samples,
                                                                       num_process=self.workflow.opt_thread_num)
            all_samples = filter_batch_output.returned_batch
        return cost / all_samples_num

    def __str__(self):
        return "workflow = " + str(self.workflow) + "\tmin_cost = " + str(self.min_cost) + "\tmax_cost = " + \
               str(self.max_cost) + "\tcurrent_node = " + str(self.node)

    def detail_info(self):
        """
        the detail information of the candidate_workflow
        """
        print("\tworkflow = " + str(self.workflow) + "\tmin_cost = " + str(self.min_cost) + "\tmax_cost = " +
              str(self.max_cost) + "flag = " + str(self.flag))
        print(
            "\tmin_selectivity = " + str(self.min_selectivities) + "\tmax_selectivity = " + str(self.max_selectivities))
        print("\tmin_reduction = " + str(self.min_reductions) + "\tmax_reduction = " + str(self.max_reductions))
        print("\tmin_pp_selectivity = " + str(self.min_pp_selectivities) + "\tmax_pp_selectivity = " +
              str(self.max_pp_selectivities))
        print("\tmin_accuracy = " + str(self.min_accuracy) + "\tmax_accuracy = " + str(self.max_accuracy))

    __repr__ = __str__


"""
utility methods
"""


def reset_scan(workflow):
    """
    reset the scan to the beginning
    """
    scan, sink = workflow.get_scan_sink()
    scan.reset_to_beginning()


def print_tree(root_node: MyNode):
    """
    print a tree starts from the root_node
    """
    for pre, fill, node in RenderTree(root_node):
        tree_str = u"%s%s" % (pre, node)
        print(tree_str.ljust(8))


def print_workflow_candidates(workflow_candidates: List[WorkflowCandidate]):
    """
    format printing workflow_candidates
    """
    for workflow_candidate in workflow_candidates:
        print("\t" + str(workflow_candidate))


def print_workflow_candiates_details(workflow_candidates: List[WorkflowCandidate]):
    """
    format printing workflow_candidates detail information
    """
    print("workflow_candidates detail information:")
    for workflow_candidate in workflow_candidates:
        workflow_candidate.detail_info()


def print_nodes(nodes: List[MyNode]):
    """
    format printing a list of nodes
    """
    print("start printing nodes list:")
    for node in nodes:
        print(node)
    print("finish printing nodes list.")
