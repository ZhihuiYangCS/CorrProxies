import time
import rootpath

rootpath.append()
from utility.constant import UCF101_TYPE
from optimizer.accuracy_allocator.pp_manager_base import PPManagerBase
from utility.utility import sample_selectivity_stop_condition, get_selectivity, sample_train_stop_condition, \
    copy_samples, train_validate_test_split
from optimizer.reorder.reorder_utility import MyNode, WorkflowCandidate, print_workflow_candidates
from optimizer.reorder.reorder.tree.st_tree.fine_grained_st_tree import FineGrainedSTTree


class FinedGrainedBBP(FineGrainedSTTree):
    """
    This class implements the branch-and-bound pruning method on the fine-grained S-T search tree
    """

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
        # compute selectivity, min_pp_selectivity and max_pp_selectivity
        selectivity = s_node.pass_num / s_node.process_num
        match_workflows = self._get_node_match_candidate_workflows(node=s_node)
        path_s_nodes, path_t_nodes = s_node.get_pass_st_nodes()  # not include s_node itself
        index = len(path_s_nodes)
        if len(path_t_nodes) == 0:
            min_pp_selectivity = selectivity
            max_pp_selectivity = selectivity
            no_t_count = index
        else:
            max_labeled_samples = s_node.labeled_samples.copy()
            min_labeled_samples = max_labeled_samples.copy()
            no_t_count = index - len(path_t_nodes)
            for path_t_node in path_t_nodes:
                if path_t_node.max_pp_filter is not None:
                    if self.workflow.workflow_type == UCF101_TYPE and self.workflow.opt_thread_num == 0:
                        max_pp_batch_output = path_t_node.max_pp_filter.process_batch(batch=max_labeled_samples,
                                                                                      model=self.workflow.models[
                                                                                          self.workflow.preprocessor.operator_name])
                        max_labeled_samples = max_pp_batch_output.returned_batch
                        min_pp_batch_output = path_t_node.min_pp_filter.process_batch(batch=min_labeled_samples,
                                                                                      model=self.workflow.models[
                                                                                          self.workflow.preprocessor.operator_name])
                        min_labeled_samples = min_pp_batch_output.returned_batch
                    else:
                        max_pp_batch_output = path_t_node.max_pp_filter.multiple_threads_batch(
                            batch=max_labeled_samples,
                            num_process=self.workflow.opt_thread_num)
                        max_labeled_samples = max_pp_batch_output.returned_batch
                        min_pp_batch_output = path_t_node.min_pp_filter.multiple_threads_batch(
                            batch=min_labeled_samples,
                            num_process=self.workflow.opt_thread_num)
                        min_labeled_samples = min_pp_batch_output.returned_batch
                else:
                    no_t_count += 1
            max_pp_selectivity = get_selectivity(max_labeled_samples)
            min_pp_selectivity = get_selectivity(min_labeled_samples)
        # update match workflows
        self._S_update_workflow_selectivity(match_workflows=match_workflows, index=index, selectivity=selectivity,
                                            no_t_count=no_t_count, min_pp_selectivity=min_pp_selectivity,
                                            max_pp_selectivity=max_pp_selectivity)
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

    def _t_node_propagation(self, corresponding_s_node: MyNode, t_node: MyNode):
        """
        propagating this t_node's pp_manager, min_pp_filter and max_pp_filter to same t_nodes
        :param corresponding_s_node: the corresponding s_node of this t_node.
                                     It also is the root_node to search for the same t_nodes
        :param t_node: the t_node to propagate
        """
        children_nodes = list(corresponding_s_node.children)
        while children_nodes:
            if children_nodes[0].node_type == "T" and children_nodes[
                0].operator.operator_name == corresponding_s_node.operator.operator_name and children_nodes[
                0].max_pp_filter is None:
                children_nodes[0].max_pp_filter = t_node.max_pp_filter
                children_nodes[0].min_pp_filter = t_node.min_pp_filter
                children_nodes[0].pp_manager = t_node.pp_manager
                children_nodes.pop(0)
            else:
                new_children_nodes = list(children_nodes[0].children)
                children_nodes.pop(0)
                children_nodes += new_children_nodes

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
            max_labeled_data = s_node.labeled_samples.copy()
            min_labeled_data = max_labeled_data.copy()
            for path_t_node in path_t_nodes:
                if self.workflow.workflow_type == UCF101_TYPE and self.workflow.opt_thread_num == 0:
                    max_pp_batch_output = path_t_node.max_pp_filter.process_batch(batch=max_labeled_data,
                                                                                  model=self.workflow.models[
                                                                                      self.workflow.preprocessor.operator_name])
                    max_labeled_data = max_pp_batch_output.returned_batch
                    min_pp_batch_output = path_t_node.min_pp_filter.process_batch(batch=min_labeled_data,
                                                                                  model=self.workflow.models[
                                                                                      self.workflow.preprocessor.operator_name])
                    min_labeled_data = min_pp_batch_output.returned_batch
                else:
                    max_pp_batch_output = path_t_node.max_pp_filter.multiple_threads_batch(batch=max_labeled_data,
                                                                                           num_process=self.workflow.opt_thread_num)
                    max_labeled_data = max_pp_batch_output.returned_batch
                    min_pp_batch_output = path_t_node.min_pp_filter.multiple_threads_batch(batch=min_labeled_data,
                                                                                           num_process=self.workflow.opt_thread_num)
                    min_labeled_data = min_pp_batch_output.returned_batch
            max_pp_selectivity = get_selectivity(max_labeled_data)
            min_pp_selectivity = get_selectivity(min_labeled_data)
            index = len(s_path_s_nodes)
            for i in range(no_t_count):
                min_pp_selectivity = min_pp_selectivity / self.workflow.target_accuracy - (
                        1 - self.workflow.target_accuracy) / self.workflow.target_accuracy
                max_pp_selectivity = max_pp_selectivity / self.workflow.target_accuracy
            candidate_workflow.min_pp_selectivities[index] = min_pp_selectivity
            candidate_workflow.max_pp_selectivities[index] = max_pp_selectivity

    def _T_node_action(self, t_node: MyNode):
        """
        If the t_node is None, it has not been propagated.
        1. initialize t_node.pp_manage
        2. compute corresponding s_node of the t_node to generate labeled_samples for training
        3. all upstream t_nodes' pp_filter process this labeled samples
        4. train max_pp_filter and min_pp_filter
        5. t_node propagation
        6. get matching candidate_workflows
        7. update max_reduction, min_reduction and max_pp_selectivity, min_pp_selectivity
        :param t_node: the t_node to take action
        """
        path_s_nodes, path_t_nodes = t_node.get_pass_st_nodes()
        if t_node.max_pp_filter is None:
            # initialize t_node.pp_manager
            t_node.pp_manager = PPManagerBase(workflow=self.workflow)
            # get the t_node's corresponding s_node and generate labeled_samples for training
            corresponding_s_node = path_s_nodes[len(path_t_nodes)]
            time1 = time.time()
            self._generate_S_node_labeled_samples_batch_process(node=corresponding_s_node,
                                                                predicate=sample_train_stop_condition)
            time2 = time.time()
            self.label_time += (time2 - time1)
            corresponding_s_node.labeled_samples = copy_samples(workflow_type=self.workflow.workflow_type,
                                                                sample=corresponding_s_node.labeled_samples)
            train_validate_test_split(sample=corresponding_s_node.labeled_samples)
            max_labeled_samples = corresponding_s_node.labeled_samples.copy()
            min_labeled_samples = max_labeled_samples.copy()
            for path_t_node in path_t_nodes:
                if path_t_node.min_pp_filter is not None:
                    if self.workflow.workflow_type == UCF101_TYPE and self.workflow.opt_thread_num == 0:
                        max_pp_batch_output = path_t_node.max_pp_filter.process_batch(batch=max_labeled_samples,
                                                                                      model=self.workflow.models[
                                                                                          self.workflow.preprocessor.operator_name])
                        max_labeled_samples = max_pp_batch_output.returned_batch
                        min_pp_batch_output = path_t_node.min_pp_filter.process_batch(batch=min_labeled_samples,
                                                                                      model=self.workflow.models[
                                                                                          self.workflow.preprocessor.operator_name])
                        min_labeled_samples = min_pp_batch_output.returned_batch
                    else:
                        max_pp_batch_output = path_t_node.max_pp_filter.multiple_threads_batch(
                            batch=max_labeled_samples,
                            num_process=self.workflow.opt_thread_num)
                        max_labeled_samples = max_pp_batch_output.returned_batch
                        min_pp_batch_output = path_t_node.min_pp_filter.multiple_threads_batch(
                            batch=min_labeled_samples,
                            num_process=self.workflow.opt_thread_num)
                        min_labeled_samples = min_pp_batch_output.returned_batch
            time3 = time.time()
            t_node.max_pp_filter = self._t_node_train_pp(t_node=t_node, data_set=max_labeled_samples, accuracy=1.0)
            t_node.min_pp_filter = self._t_node_train_pp(t_node=t_node, data_set=min_labeled_samples,
                                                         accuracy=self.workflow.target_accuracy)
            time4 = time.time()
            self.train_time += (time4 - time3)
            self._t_node_propagation(corresponding_s_node=corresponding_s_node, t_node=t_node)
        match_workflows = self._get_node_match_candidate_workflows(node=t_node)
        index = len(path_t_nodes)
        for workflow in match_workflows:
            workflow.flag = True
            workflow.max_reductions[index] = t_node.max_pp_filter.accuracy_info_dict.reduction
            workflow.min_reductions[index] = t_node.min_pp_filter.accuracy_info_dict.reduction
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
                self._T_node_action(t_node=candidate_node)
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

        print("******** after reorder search *********")
        print_workflow_candidates(workflow_candidates=self.candidate_workflows)

        print("******** allocate accuracy *********")
        if len(self.candidate_workflows) == 1:
            self.find_optimal_order_time += (time.time() - time1)
            children_nodes = self._candidate_workflow_children_nodes(candidate_workflow=self.candidate_workflows[0])
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
                children_nodes = self._candidate_workflow_children_nodes(
                    candidate_workflow=self.candidate_workflows[0])
            self._one_workflow_allocate_accuracy(candidate_workflow=self.candidate_workflows[0])
        else:
            while len(self.candidate_workflows) > 1:
                self._one_workflow_allocate_accuracy(self.candidate_workflows[0])
                self.candidate_workflow_sort()
                print("\tcandidate workflows after sort")
                print_workflow_candidates(workflow_candidates=self.candidate_workflows)
            self.find_optimal_order_time += (time.time() - time1)
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
