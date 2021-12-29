import time
from typing import List
import rootpath

rootpath.append()
from utility.constant import UCF101_TYPE
from optimizer.accuracy_allocator.pp_manager_base import PPManagerBase
from utility.utility import sample_selectivity_stop_condition, get_selectivity, copy_samples, \
    sample_train_stop_condition, train_validate_test_split
from optimizer.reorder.reorder_utility import MyNode, print_workflow_candidates, WorkflowCandidate
from optimizer.reorder.reorder.tree.st_tree.fine_grained_st_tree import FineGrainedSTTree


class FineGrainedAA(FineGrainedSTTree):
    def __init__(self, workflow):
        super().__init__(workflow=workflow)

    def _S_node_action(self, s_node: MyNode):
        """
        The action of s_node;
        1. generate labeled_samples to get selectivity
        2. if it is needed, update ml_operator cost
        3. get all matching candidate_workflows
        4. update matching candidate_workflows:
                (min_selectivity = selectivity, max_selectivity = selectivity;
                 min_pp_selectivity = selectivity/target_accuracy - (1-target_accuracy)/target_accuracy,
                 max_pp_selectivity = selectivity/target_accuracy
                 min_reduction = 1-min_pp_selectivity*target_accuracy)
        5. AB -> BA selectivity
        :param s_node: the s_node to take action
        """
        # generate labeled_samples
        time1 = time.time()
        self._generate_S_node_labeled_samples_batch_process(node=s_node,
                                                            predicate=sample_selectivity_stop_condition)
        time2 = time.time()
        self.label_time += (time2 - time1)
        # update constant_cost
        if self.ml_operator_cost_update:
            update_operators = self.constant_cost.check_update()
            if update_operators:
                operator_name = s_node.operator.operator_name
                if operator_name in update_operators:
                    self.constant_cost.flag = True
                    s_node.operator.get_operator_cost()
                    self.constant_cost.update_operator_cost(operator_name=operator_name,
                                                            cost=s_node.operator.operator_cost)
                print("\t" + str(self.constant_cost))
            else:
                self.ml_operator_cost_update = False
        # compute selectivity, pp_selectivity
        selectivity = s_node.pass_num / s_node.process_num
        match_workflows = self._get_node_match_candidate_workflows(node=s_node)
        path_s_nodes, path_t_nodes = s_node.get_pass_st_nodes()  # not include s_node itself
        index = len(path_s_nodes)
        if len(path_t_nodes) == 0:
            pp_selectivity = selectivity
            no_t_count = index
        else:
            labeled_sample = s_node.labeled_samples.copy()
            no_t_count = index - len(path_t_nodes)
            for path_t_node in path_t_nodes:
                if path_t_node.max_pp_filter is not None:
                    if self.workflow.workflow_type == UCF101_TYPE and self.workflow.opt_thread_num == 0:
                        pp_batch_output = path_t_node.max_pp_filter.process_batch(batch=labeled_sample,
                                                                                  model=self.workflow.models[
                                                                                      self.workflow.preprocessor.operator_name])
                    else:
                        pp_batch_output = path_t_node.max_pp_filter.multiple_threads_batch(batch=labeled_sample,
                                                                                           num_process=self.workflow.opt_thread_num)
                    labeled_sample = pp_batch_output.returned_batch
                else:
                    no_t_count += 1
            pp_selectivity = get_selectivity(labeled_sample)
        # update match workflows
        self._S_update_workflow_selectivity(match_workflows=match_workflows, index=index, selectivity=selectivity,
                                            no_t_count=no_t_count, min_pp_selectivity=pp_selectivity,
                                            max_pp_selectivity=pp_selectivity)
        # update AB -> BA selectivity
        if len(path_s_nodes) >= 1:
            multiply_selectivity = match_workflows[0].min_selectivities[index] * match_workflows[0].min_selectivities[
                index - 1]
            last_nodes = path_s_nodes[-1]
            path_s_nodes[-1] = s_node
            path_s_nodes.append(last_nodes)
            new_match_workflows = []
            for workflow in self.candidate_workflows:
                if self._pass_nodes_equals_candidate_workflow(candidate_workflow=workflow, nodes=path_s_nodes):
                    new_match_workflows.append(workflow)
            for workflow in new_match_workflows:
                if workflow.min_selectivities[index - 1] != 0:
                    new_selectivity = multiply_selectivity / workflow.min_selectivities[index - 1]
                    """
                    no_t_count = index, because we don't get the labeled_sample. 
                    we can't use previous trained PPs, if it has, to filter the labeled sample, 
                    to compute pp_selectivity
                    """
                    self._S_update_workflow_selectivity(match_workflows=[workflow], index=index,
                                                        selectivity=new_selectivity, no_t_count=index,
                                                        min_pp_selectivity=new_selectivity,
                                                        max_pp_selectivity=new_selectivity)

    def _t_nodes_propagation(self, t_nodes: List[MyNode]):
        """
        1. get all leaves
        2. for each leaves get path_t_nodes.
        3. If path_t_nodes match t_nodes, update path_t_nodes
        :param t_nodes: a list of MyNode
        """
        children_nodes = list(self.scan_root.leaves)
        for child_node in children_nodes:
            _, path_t_nodes = child_node.get_pass_st_nodes()
            if child_node.node_type == "T":
                path_t_nodes.append(child_node)
            flag = True
            for index, t_node in enumerate(t_nodes):
                if t_node.operator.operator_name != path_t_nodes[index].operator.operator_name:
                    flag = False
                    break
            if flag:
                for index, t_node in enumerate(t_nodes):
                    path_t_nodes[index].pp_manager = t_node.pp_manager
                    path_t_nodes[index].max_pp_filter = t_node.max_pp_filter
                    path_t_nodes[index].min_pp_filter = t_node.min_pp_filter

    def _T_update_matched_workflow_pp_selectivity(self, candidate_workflow: WorkflowCandidate, t_node: MyNode):
        """
        update the pp_selectivity for all s_nodes between the t_node's corresponding_s_node and the t_node.
        :param candidate_workflow: the candidate_workflow to update
        """
        path_s_nodes, path_t_nodes = t_node.get_pass_st_nodes()
        path_t_nodes.append(t_node)
        # get candidate_workflow's (cw) path_s_nodes and path_t_nodes
        cw_s_nodes, cw_t_nodes = candidate_workflow.node.get_pass_st_nodes()
        if candidate_workflow.node.node_type == "S":
            cw_s_nodes.append(candidate_workflow.node)
        # get all s_nodes between the t_node's corresponding s_node to the end
        s_nodes_between_t_parent_s_to_t = cw_s_nodes[len(path_t_nodes):]
        for s_node in s_nodes_between_t_parent_s_to_t:
            s_path_s_nodes, _ = s_node.get_pass_st_nodes()
            no_t_count = len(s_path_s_nodes) - len(path_t_nodes)
            labeled_data = s_node.labeled_samples.copy()
            for path_t_node in path_t_nodes:
                if self.workflow.workflow_type == UCF101_TYPE and self.workflow.opt_thread_num == 0:
                    t_batch_output = path_t_node.max_pp_filter.process_batch(batch=labeled_data,
                                                                             model=self.workflow.models[
                                                                                 self.workflow.preprocessor.operator_name])
                else:
                    t_batch_output = path_t_node.max_pp_filter.multiple_threads_batch(batch=labeled_data,
                                                                                      num_process=self.workflow.opt_thread_num)
                labeled_data = t_batch_output.returned_batch
            pp_selectivity = get_selectivity(labeled_data)
            index = len(s_path_s_nodes)
            min_pp_selectivity = max_pp_selectivity = pp_selectivity
            for i in range(no_t_count):
                min_pp_selectivity = min_pp_selectivity / self.workflow.target_accuracy - (
                        1 - self.workflow.target_accuracy) / self.workflow.target_accuracy
                max_pp_selectivity = max_pp_selectivity / self.workflow.target_accuracy
            candidate_workflow.min_pp_selectivities[index] = min_pp_selectivity
            candidate_workflow.max_pp_selectivities[index] = max_pp_selectivity

    def _T_node_allocate_accuracy(self, t_node: MyNode, candidate_workflow: WorkflowCandidate):
        """
        If t_node.max_pp_filter is None,
            initialize pp_manager,
            generate enough labeled_samples for its corresponding s_nodes
        1. allocate_accuracy for path_t_nodes, including t_node itself.
        2. propagation these path_t_nodes
        3. update matching candidate_workflows
        :param t_node: the t_node to take action
        :param candidate_workflow: the candidate_workflow that searched on.
        """
        path_s_nodes, path_t_nodes = t_node.get_pass_st_nodes()
        # get the t_node's corresponding s_node and generate labeled_samples for training
        corresponding_s_node = path_s_nodes[len(path_t_nodes)]
        # if not None, this means this t_node and its corresponding s_node has been explored.
        # The t_node has pp_manager. The t_node's corresponding s_node has enough labeled_samples
        if t_node.max_pp_filter is None:
            # initialize t_node.pp_manager
            t_node.pp_manager = PPManagerBase(workflow=self.workflow)
            time1 = time.time()
            self._generate_S_node_labeled_samples_batch_process(node=corresponding_s_node,
                                                                predicate=sample_train_stop_condition)
            time2 = time.time()
            self.label_time += (time2 - time1)
            corresponding_s_node.labeled_samples = copy_samples(workflow_type=self.workflow.workflow_type,
                                                                sample=corresponding_s_node.labeled_samples)
            train_validate_test_split(sample=corresponding_s_node.labeled_samples)
        path_t_nodes.append(t_node)
        self._s_t_nodes_allocate_accuracy(s_nodes=path_s_nodes, t_nodes=path_t_nodes,
                                          candidate_workflow=candidate_workflow)
        self._t_nodes_propagation(t_nodes=path_t_nodes)
        match_workflows = self._get_node_match_candidate_workflows(node=t_node)
        for workflow in match_workflows:
            if str(workflow.workflow.workflow) != str(candidate_workflow.workflow.workflow):
                workflow.flag = True
                workflow.max_reductions = candidate_workflow.max_reductions.copy()
                workflow.min_reductions = candidate_workflow.min_reductions.copy()
                workflow.max_accuracy = candidate_workflow.max_accuracy.copy()
                workflow.min_accuracy = candidate_workflow.min_accuracy.copy()
                self._T_update_matched_workflow_pp_selectivity(candidate_workflow=workflow, t_node=t_node)

    def search(self):
        """
        search on the generated tree.
        """
        time1 = time.time()
        execution = 0
        index = self._get_candidate_workflow_index()
        while index is not None:
            candidate_nodes = self._candidate_workflow_children_nodes(
                candidate_workflow=self.candidate_workflows[index])
            print("*******the " + str(execution) + "th search*******")
            print("\tselected workflow: " + str(self.candidate_workflows[index]))
            print("\tcandidate_nodes = " + str(candidate_nodes))
            if len(candidate_nodes) == 1:
                candidate_node = candidate_nodes[0]
            else:
                candidate_node = self._T_S_nodes_competition(candidate_workflow=self.candidate_workflows[index],
                                                             candidate_nodes=candidate_nodes)
            if candidate_node.node_type == "S":
                self._S_node_action(s_node=candidate_node)
            else:
                self._T_node_allocate_accuracy(t_node=candidate_node,
                                               candidate_workflow=self.candidate_workflows[index])
            self._update_workflow_node(node=candidate_node)
            self._update_workflow_cost()
            print("\tworkflow with updated cost")
            print_workflow_candidates(workflow_candidates=self.candidate_workflows)
            self.candidate_workflow_sort()
            print("\tcandidate workflows after sort")
            print_workflow_candidates(workflow_candidates=self.candidate_workflows)
            execution += 1
            index = self._get_candidate_workflow_index()
        time2 = time.time()
        self.find_optimal_order_time += (time.time() - time1)
        children_nodes = self._candidate_workflow_children_nodes(candidate_workflow=self.candidate_workflows[0])
        print("children_nodes = " + str(children_nodes))
        while children_nodes:
            if len(children_nodes) == 1:
                candidate_node = children_nodes[0]
            else:
                candidate_node = self._T_S_competition_selected_workflow(candidate_nodes=children_nodes)
            if candidate_node.node_type == "S":
                self._S_node_selected_workflow(s_node=candidate_node,
                                               candidate_workflow=self.candidate_workflows[0])
            else:
                self._T_node_selected_workflow(t_node=candidate_node,
                                               candidate_workflow=self.candidate_workflows[0])
            self.candidate_workflows[0].node = candidate_node
            children_nodes = self._candidate_workflow_children_nodes(candidate_workflow=self.candidate_workflows[0])
        self._one_workflow_allocate_accuracy(candidate_workflow=self.candidate_workflows[0])

        print("\t all candidate_workflows:")
        print_workflow_candidates(self.candidate_workflows)
        pp_filters, ml_filters = self._selected_workflow(candidate_workflow=self.candidate_workflows[0])
        time3 = time.time()
        print("reorder search time = " + str(time2 - time1))
        print("allocate accuracy time = " + str(time3 - time2))
        self.reorder_time += (time3 - time1)
        self.train_time += self.accuracy_allocator.train_time
        self.aa_time = self.aa_time - self.accuracy_allocator.train_time
        time_log = str(self.label_time) + "\t" + str(self.train_time) + "\t" + str(self.aa_time) + "\t" + str(
            self.reorder_time) + "\t" + str(self.find_optimal_order_time)
        return pp_filters, ml_filters, time_log
