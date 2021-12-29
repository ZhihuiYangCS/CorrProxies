import time
from abc import ABC
from typing import List
from anytree.exporter import DotExporter
import rootpath

rootpath.append()
from optimizer.accuracy_allocator.pp_manager_base import PPManagerBase
from utility.utility import sample_selectivity_stop_condition, sample_train_stop_condition, \
    copy_samples, train_validate_test_split
from optimizer.reorder.reorder.tree.reorder_tree import ReorderTree
from optimizer.reorder.reorder_utility import MyNode, WorkflowCandidate, print_workflow_candidates, print_tree
from utility.constant import TWITTER_SAMPLE_SIZE, COCO_SAMPLE_SIZE, UCF101_SAMPLE_SIZE, TWITTER_SELECT_SIZE, \
    COCO_SELECT_SIZE, UCF101_SELECT_SIZE


class FineGrainedSTTree(ReorderTree, ABC):
    """
    an abstract for the search method based on fine grained S-T tree
        generate_search_tree
    """

    def __init__(self, workflow):
        super().__init__(workflow=workflow)
        if self.workflow.workflow_type == 0:
            self.train_size = TWITTER_SAMPLE_SIZE
            self.sample_size = TWITTER_SELECT_SIZE
        elif self.workflow.workflow_type == 1:
            self.train_size = COCO_SAMPLE_SIZE
            self.sample_size = COCO_SELECT_SIZE
        else:
            self.train_size = UCF101_SAMPLE_SIZE
            self.sample_size = UCF101_SELECT_SIZE

    def generate_search_tree(self):
        """
        generate a search tree, fine grained S-T search tree
        """
        time1 = time.time()
        operator_index = 1
        current_leaves = [self.scan_root]
        for i in range(len(self.ml_filters) * 2):
            nodes = []
            for node in current_leaves:
                operator_index, t_nodes = self._generate_first_t_node(node=node, operator_index=operator_index)
                operator_index, s_nodes = self._generate_all_s_nodes(node=node, operator_index=operator_index)
                nodes.extend(t_nodes)
                nodes.extend(s_nodes)
            current_leaves = nodes.copy()
        time2 = time.time()
        print("generate tree time = " + str((time2 - time1)))
        # DotExporter(self.scan_root).to_picture("fine_grained_st_tree.png")
        # print_tree(self.scan_root)

    def _S_go_back_get_data_cost(self, path_s_nodes: List[MyNode], num: int,
                                 candidate_workflow: WorkflowCandidate) -> float:
        """
        This method estimate the cost of getting data for a s_node
        """
        cost = 0
        for i in range(len(path_s_nodes) - 1, -1, -1):
            if len(path_s_nodes[i].cache) != 0:
                if num >= len(path_s_nodes[i].cache):
                    selectivity = 1
                    for j in range(i, len(path_s_nodes)):
                        cost += selectivity * len(path_s_nodes[i].cache) * self.constant_cost.min_costs[
                            self.constant_cost.operator_names.index(path_s_nodes[j].operator.operator_name)]
                        if j != len(path_s_nodes) - 1:
                            selectivity *= candidate_workflow.min_selectivities[j]
                    num -= len(path_s_nodes[i].cache)
                else:
                    selectivity = 1
                    for j in range(i, len(path_s_nodes)):
                        cost += selectivity * num * self.constant_cost.min_costs[
                            self.constant_cost.operator_names.index(path_s_nodes[j].operator.operator_name)]
                        if j != len(path_s_nodes) - 1:
                            selectivity *= candidate_workflow.min_selectivities[j]
                    num = 0
            if num <= 0:
                break
        if num > 0:
            for i in range(len(path_s_nodes) - 2, -1, -1):
                num /= candidate_workflow.min_selectivities[i]
            selectivity = 1
            for i in range(0, len(path_s_nodes)):
                cost += selectivity * num * self.constant_cost.min_costs[self.constant_cost.operator_names.index(
                    path_s_nodes[i].operator.operator_name)]
                if i != len(path_s_nodes) - 1:
                    selectivity *= candidate_workflow.min_selectivities[i]
        # it's not linear.
        # because both of T-node and S-node to get labeled samples need call this function to estimate the cost,
        # it's ok to ignore the opt_thread_num. They need to divide the same number.
        # cost = cost/self.workflow.opt_thread_num
        return cost

    def _T_S_nodes_competition(self, candidate_workflow: WorkflowCandidate, candidate_nodes: List[MyNode]) -> MyNode:
        """
        candidate t_node and s_node competition
        :param candidate_workflow: the candidate workflow to explore
        :param candidate_nodes:
        :return:
        """
        return_node_type = None
        node_type_node_dict = {}
        for node in candidate_nodes:
            node_type_node_dict[node.node_type] = node
            if node.node_type == "T" and node.max_pp_filter is not None:
                return_node_type = "S"
            if node.node_type == "S":
                path_s_nodes, path_t_nodes = node.get_pass_st_nodes()
                index = len(path_s_nodes)
                if candidate_workflow.min_selectivities[index] == candidate_workflow.max_selectivities[index]:
                    return_node_type = "T"
        if return_node_type is not None:
            return node_type_node_dict[return_node_type]
        cost_list, benefit_list = [], []
        for node in candidate_nodes:
            path_s_nodes, path_t_nodes = node.get_pass_st_nodes()
            if node.node_type == "T":
                pass_corresponding_s_nodes = path_s_nodes[0:len(path_t_nodes) + 1]
                num = self.train_size - len(pass_corresponding_s_nodes[-1].labeled_samples)
                cost = self._S_go_back_get_data_cost(path_s_nodes=pass_corresponding_s_nodes, num=num,
                                                     candidate_workflow=candidate_workflow)
                cost += self.constant_cost.train_cost
                cost_list.append(cost)
                # compute benefit: S_1' * a_1 * S_2' * a_2 (r_3(a_3)C_3 - C_c)
                min_pp_selectivity, max_pp_selectivity = 1, 1
                for i in range(len(path_t_nodes)):
                    min_pp_selectivity *= candidate_workflow.min_pp_selectivities[i] * self.workflow.target_accuracy
                    max_pp_selectivity *= candidate_workflow.max_pp_selectivities[i]
                max_reduction = candidate_workflow.max_reductions[len(path_t_nodes)] * self.constant_cost.min_costs[
                    self.constant_cost.operator_names.index(
                        node.operator.operator_name)] - self.constant_cost.infer_cost
                if max_reduction < 0:
                    max_reduction = 0
                min_reduction = candidate_workflow.min_reductions[len(path_t_nodes)] * self.constant_cost.min_costs[
                    self.constant_cost.operator_names.index(
                        node.operator.operator_name)] - self.constant_cost.infer_cost
                node_benefit = -(max_pp_selectivity * max_reduction - min_pp_selectivity * min_reduction)
                benefit_list.append(node_benefit)
            elif node.node_type == "S":
                path_s_nodes.append(node)
                num = self.sample_size - len(path_s_nodes[-1].labeled_samples)
                cost = self._S_go_back_get_data_cost(path_s_nodes=path_s_nodes, num=num,
                                                     candidate_workflow=candidate_workflow)
                cost_list.append(cost)
                # compute benefit
                selectivity = 1
                for i in range(len(path_s_nodes) - 1):
                    selectivity *= candidate_workflow.min_selectivities[i]
                max_selectivity = selectivity * candidate_workflow.max_selectivities[len(path_s_nodes) - 1]
                min_selectivity = selectivity * candidate_workflow.min_selectivities[len(path_s_nodes) - 1]
                max_benefit = 0
                min_benefit = 0
                for i in range(len(path_s_nodes), candidate_workflow.ml_filters_num):
                    operator_cost = self.constant_cost.min_costs[self.constant_cost.operator_names.index(
                        candidate_workflow.ml_filters[i].operator_name)]
                    max_benefit += max_selectivity * operator_cost
                    max_selectivity *= candidate_workflow.max_selectivities[i]
                    min_benefit += min_selectivity * operator_cost
                    min_selectivity *= candidate_workflow.min_selectivities[i]
                benefit_list.append((max_benefit - min_benefit))
        print("\tcandidate_nodes = " + str(candidate_nodes) + "\tcosts = " + str(cost_list) + "\tbenefit = " + str(
            benefit_list))
        max_benefit = 0
        index = 0
        for i in range(len(cost_list)):
            benefit_cost = benefit_list[i] / cost_list[i]
            if benefit_cost > max_benefit:
                max_benefit = benefit_cost
                index = i
        print("\tselected node = " + str(candidate_nodes[index]))
        return candidate_nodes[index]

    def _s_t_nodes_allocate_accuracy(self, s_nodes: List[MyNode], t_nodes: List[MyNode],
                                     candidate_workflow: WorkflowCandidate):
        labeled_samples, pp_managers = [], []
        for s_node in s_nodes:
            labeled_samples.append(s_node.labeled_samples.copy())
        for t_node in t_nodes:
            pp_managers.append(t_node.pp_manager)
        time1 = time.time()
        if len(self.candidate_workflows) == 1:
            pp_filters, _, train_time = self.accuracy_allocator.allocate_accuracy(
                labeled_samples=labeled_samples, pp_managers=pp_managers, candidate_workflow=candidate_workflow,
                first_node_input=s_nodes[0].first_s_node_input)
            # self.train_time += train_time
        else:
            pp_filters, _, train_time = self.accuracy_allocator.allocate_accuracy(
                labeled_samples=labeled_samples, pp_managers=pp_managers, candidate_workflow=candidate_workflow,
                final_validation_flag=False, first_node_input=s_nodes[0].first_s_node_input)
            # self.train_time += train_time
        time2 = time.time()
        self.aa_time += (time2 - time1)
        print("\t all candidate_workflows:")
        print_workflow_candidates(self.candidate_workflows)
        for index, t_node in enumerate(t_nodes):
            t_node.max_pp_filter = t_node.min_pp_filter = pp_filters[index]

    def _one_workflow_allocate_accuracy(self, candidate_workflow: WorkflowCandidate):
        """
        Perform accuracy allocation for the selected workflow.
        First get all s_nodes and t_nodes for the selected workflow
        Then get all labeled_samples from s_nodes, and pp_managers from t_nodes.
        Next, perform accuracy allocation to get pp_filters
        Finally, add pp_filters to all t_nodes
        :param candidate_workflow: the selected workflow to perform accuracy allocation
        """
        s_nodes, t_nodes = candidate_workflow.node.get_pass_st_nodes()
        if candidate_workflow.node.node_type == "S":
            s_nodes.append(candidate_workflow.node)
        if candidate_workflow.node.node_type == "T":
            t_nodes.append(candidate_workflow.node)
        self._s_t_nodes_allocate_accuracy(s_nodes=s_nodes, t_nodes=t_nodes, candidate_workflow=candidate_workflow)

    def _T_S_competition_selected_workflow(self, candidate_nodes: List[MyNode]):
        """
        In selected workflow, there is no need to use heuristic method for T-node and S-node competition.
        We only return the T-node.
        :param candidate_nodes: a list of candidate_nodes
        :return: a T-node
        """
        result = None
        for node in candidate_nodes:
            if node.node_type == "T":
                result = node
        return result

    def _selected_workflow(self, candidate_workflow: WorkflowCandidate):
        """
        get pp_filters and ml_filters for the selected workflow.
        This method is used when the selected workflow has already been allocated accuracy.
        :param candidate_workflow: the selected workflow
        :return: a list of pp_filters and a list of ml_filters.
        """
        s_nodes, t_nodes = candidate_workflow.node.get_pass_st_nodes()
        if candidate_workflow.node.node_type == "S":
            s_nodes.append(candidate_workflow.node)
        if candidate_workflow.node.node_type == "T":
            t_nodes.append(candidate_workflow.node)
        pp_filters, ml_filters = [], []
        for t_node in t_nodes:
            pp_filters.append(t_node.max_pp_filter)
            ml_filters.append(t_node.operator)
        return pp_filters, ml_filters

    def _S_node_selected_workflow(self, s_node: MyNode, candidate_workflow: WorkflowCandidate):
        """
        This method is used when we have get one selected workflow,
        and the selected workflow still has some s-nodes to execute.
        The action for the s-node is only generate labeled-samples and update selected workflow's selectivity.
        The selectivity is used in accuracy allocation for the selected workflow.
        :param s_node: the s_node to take action
        :param candidate_workflow: the selected workflow
        """
        time1 = time.time()
        self._generate_S_node_labeled_samples_batch_process(node=s_node,
                                                            predicate=sample_selectivity_stop_condition)
        time2 = time.time()
        self.label_time += (time2 - time1)
        selectivity = s_node.pass_num / s_node.process_num
        path_s_nodes, path_t_nodes = s_node.get_pass_st_nodes()
        index = len(path_s_nodes)
        candidate_workflow.min_selectivities[index] = candidate_workflow.max_selectivities[index] = selectivity

    def _T_node_selected_workflow(self, t_node: MyNode, candidate_workflow: WorkflowCandidate):
        """
        This method is used when we have get one selected workflow,
        and the selected workflow still has some t-nodes to execute.
        The action for the t-node is only to initialize the pp_manager,
        and generate enough labeled_samples for training PPs.
        :param t_node: the t_node to take action
        :param candidate_workflow: the selected workflow
        """
        path_s_nodes, path_t_nodes = t_node.get_pass_st_nodes()
        if t_node.max_pp_filter is None:
            t_node.pp_manager = PPManagerBase(workflow=self.workflow)
            corresponding_s_node = path_s_nodes[len(path_t_nodes)]
            time1 = time.time()
            self._generate_S_node_labeled_samples_batch_process(node=corresponding_s_node,
                                                                predicate=sample_train_stop_condition)
            time2 = time.time()
            self.label_time += (time2 - time1)
            corresponding_s_node.labeled_samples = copy_samples(workflow_type=self.workflow.workflow_type,
                                                                sample=corresponding_s_node.labeled_samples)
            train_validate_test_split(sample=corresponding_s_node.labeled_samples)
