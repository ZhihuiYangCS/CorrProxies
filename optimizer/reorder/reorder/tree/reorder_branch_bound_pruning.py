import time
import rootpath

rootpath.append()
from utility.constant import UCF101_TYPE
from utility.utility import sample_train_stop_condition, copy_samples, train_validate_test_split, get_selectivity
from optimizer.accuracy_allocator.pp_manager_base import PPManagerBase
from optimizer.reorder.reorder_utility import MyNode, print_workflow_candidates, WorkflowCandidate
from optimizer.reorder.reorder.tree.reorder_tree import ReorderTree


class ReorderBBP(ReorderTree):
    """
    branch and bound method on an enumerating all possible plan tree
    """

    def __init__(self, workflow):
        super().__init__(workflow=workflow)

    def generate_search_tree(self):
        """
        generate a search tree, enumerate all possible plan tree
        """
        time1 = time.time()
        operator_index = 1
        current_leaves = [self.scan_root]
        for i in range(len(self.ml_filters)):
            nodes = []
            for node in current_leaves:
                operator_index, s_nodes = self._generate_all_s_nodes(node=node, operator_index=operator_index)
                nodes.extend(s_nodes)
            current_leaves = nodes.copy()
        time2 = time.time()
        print("generate tree time = " + str((time2 - time1)))
        # DotExporter(self.scan_root).to_picture("fine_grained_st_tree.png")
        # print_tree(self.scan_root)

    def _ST_node_action(self, node: MyNode):
        """
        1. initialize node.pp_manager
        2. generate enough labeled_samples, and prepare labeled_samples to train PPs
        3. update constant_cost
        4. previous nodes' pp_filter process the labeled_samples
        5. train node.max_pp_filter and node.min_pp_filter
        6. update matching candidate_workflows' min_selectivities, max_selectivities, min_pp_selectivities,
                max_pp_selectivities, min_reduction, max_reduction
        7. update AB -> BA selectivity
        :param node: a st_node to take action
        """
        path_st_nodes, _ = node.get_pass_st_nodes()
        # initialize node.pp_manager
        node.pp_manager = PPManagerBase(workflow=self.workflow)
        # generate enough labeled_samples
        time1 = time.time()
        self._generate_S_node_labeled_samples_batch_process(node=node,
                                                            predicate=sample_train_stop_condition)
        time2 = time.time()
        self.label_time += (time2 - time1)
        # update constant_cost
        if self.ml_operator_cost_update:
            update_operators = self.constant_cost.check_update()
            if update_operators:
                operator_name = node.operator.operator_name
                if operator_name in update_operators:
                    self.constant_cost.flag = True
                    node.operator.get_operator_cost()
                    self.constant_cost.update_operator_cost(operator_name=operator_name,
                                                            cost=node.operator.operator_cost)
                print("\t" + str(self.constant_cost))
            else:
                self.ml_operator_cost_update = False
        # prepare labeled_samples to train PPs
        node.labeled_samples = copy_samples(workflow_type=self.workflow.workflow_type, sample=node.labeled_samples)
        train_validate_test_split(sample=node.labeled_samples)
        # previous nodes' pp_filter process the labeled_samples
        max_labeled_samples = node.labeled_samples.copy()
        min_labeled_samples = max_labeled_samples.copy()
        selectivity = node.pass_num / node.process_num
        for st_node in path_st_nodes:
            if self.workflow.workflow_type == UCF101_TYPE and self.workflow.opt_thread_num == 0:
                max_pp_batch_output = st_node.max_pp_filter.process_batch(batch=max_labeled_samples,
                                                                          model=self.workflow.models[
                                                                              self.workflow.preprocessor.operator_name])
                max_labeled_samples = max_pp_batch_output.returned_batch
                min_pp_batch_output = st_node.min_pp_filter.process_batch(batch=min_labeled_samples,
                                                                          model=self.workflow.models[
                                                                              self.workflow.preprocessor.operator_name])
                min_labeled_samples = min_pp_batch_output.returned_batch
            else:
                max_pp_batch_output = st_node.max_pp_filter.multiple_threads_batch(batch=max_labeled_samples,
                                                                                   num_process=self.workflow.opt_thread_num)
                max_labeled_samples = max_pp_batch_output.returned_batch
                min_pp_batch_output = st_node.min_pp_filter.multiple_threads_batch(batch=min_labeled_samples,
                                                                                   num_process=self.workflow.opt_thread_num)
                min_labeled_samples = min_pp_batch_output.returned_batch
        max_pp_selectivity = get_selectivity(max_labeled_samples)
        min_pp_selectivity = get_selectivity(min_labeled_samples)
        # train node.max_pp_filter and node.min_pp_filter
        time3 = time.time()
        node.max_pp_filter = self._t_node_train_pp(t_node=node, data_set=max_labeled_samples, accuracy=1.0)
        node.min_pp_filter = self._t_node_train_pp(t_node=node, data_set=min_labeled_samples,
                                                   accuracy=self.workflow.target_accuracy)
        time4 = time.time()
        self.train_time += (time4 - time3)
        # update matching candidate_workflows
        match_workflows = self._get_node_match_candidate_workflows(node=node)
        index = len(path_st_nodes)
        for workflow in match_workflows:
            workflow.flag = True
            workflow.min_selectivities[index] = selectivity
            workflow.max_selectivities[index] = selectivity
            workflow.min_pp_selectivities[index] = min_pp_selectivity
            workflow.max_pp_selectivities[index] = max_pp_selectivity
            workflow.min_reductions[index] = node.min_pp_filter.accuracy_info_dict.reduction
            workflow.max_reductions[index] = node.max_pp_filter.accuracy_info_dict.reduction

        # update AB -> BA selectivity
        if len(path_st_nodes) >= 1:
            multiply_selectivity = match_workflows[0].min_selectivities[index] * match_workflows[0].min_selectivities[
                index - 1]
            last_nodes = path_st_nodes[-1]
            path_st_nodes[-1] = node
            path_st_nodes.append(last_nodes)
            new_match_workflows = []
            for workflow in self.candidate_workflows:
                if self._pass_nodes_equals_candidate_workflow(candidate_workflow=workflow, nodes=path_st_nodes):
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

    def _ST_node_selected_workflow(self, node: MyNode, candidate_workflow: WorkflowCandidate):
        """
        This method is used when we have get one selected workflow,
        and the selected workflow still has some st-nodes to execute.
        1. generate labeled-samples
        2. update selected workflow's selectivity.
           The selectivity is used in accuracy allocation for the selected workflow.
        3. initialize the pp_manager
        4. prepare labeled_samples for training PPs.
        :param node: the st_node to take action
        :param candidate_workflow: the selected workflow
        """
        time1 = time.time()
        self._generate_S_node_labeled_samples_batch_process(node=node,
                                                            predicate=sample_train_stop_condition)
        time2 = time.time()
        self.label_time += (time2 - time1)
        path_st_nodes, _ = node.get_pass_st_nodes()
        selectivity = node.pass_num / node.process_num
        index = len(path_st_nodes)
        candidate_workflow.min_selectivities[index] = candidate_workflow.max_selectivities[index] = selectivity
        node.pp_manager = PPManagerBase(workflow=self.workflow)
        node.labeled_samples = copy_samples(workflow_type=self.workflow.workflow_type, sample=node.labeled_samples)
        train_validate_test_split(sample=node.labeled_samples)

    def _one_workflow_allocate_accuracy(self, candidate_workflow: WorkflowCandidate):
        """
         Perform accuracy allocation for the selected workflow.
         First get all st_nodes for the selected workflow
         Then get all labeled_samples, and pp_managers from st_nodes.
         Next, perform accuracy allocation to get pp_filters
         Finally, add pp_filters to all st_nodes
         :param candidate_workflow: the selected workflow to perform accuracy allocation
         """
        st_nodes, _ = candidate_workflow.node.get_pass_st_nodes()
        st_nodes.append(candidate_workflow.node)
        labeled_samples, pp_managers = [], []
        for st_node in st_nodes:
            labeled_samples.append(st_node.labeled_samples.copy())
            pp_managers.append(st_node.pp_manager)
        print("first s_node first_s_node_input len = " + str(len(st_nodes[0].first_s_node_input)))
        time1 = time.time()
        if len(self.candidate_workflows) == 1:
            pp_filters, _, train_time = self.accuracy_allocator.allocate_accuracy(
                labeled_samples=labeled_samples, pp_managers=pp_managers, candidate_workflow=candidate_workflow,
                first_node_input=st_nodes[0].first_s_node_input)
            # self.train_time += train_time
        else:
            pp_filters, _, train_time = self.accuracy_allocator.allocate_accuracy(
                labeled_samples=labeled_samples, pp_managers=pp_managers, candidate_workflow=candidate_workflow,
                final_validation_flag=False, first_node_input=st_nodes[0].first_s_node_input)
            # self.train_time += train_time
        time2 = time.time()
        self.aa_time += (time2 - time1)
        print("\t all candidate_workflows:")
        print_workflow_candidates(self.candidate_workflows)
        for index, st_node in enumerate(st_nodes):
            st_node.max_pp_filter = st_node.min_pp_filter = pp_filters[index]

    def _selected_workflow(self, candidate_workflow: WorkflowCandidate):
        """
        get pp_filters and ml_filters for the selected workflow.
        This method is used when the selected workflow has already been allocated accuracy.
        :param candidate_workflow: the selected workflow
        :return: a list of pp_filters and a list of ml_filters.
        """
        st_nodes, _ = candidate_workflow.node.get_pass_st_nodes()
        st_nodes.append(candidate_workflow.node)
        pp_filters, ml_filters = [], []
        for st_node in st_nodes:
            pp_filters.append(st_node.max_pp_filter)
            ml_filters.append(st_node.operator)
        return pp_filters, ml_filters

    def search(self):
        """
        search on the generated tree.
        """
        time1 = time.time()
        execution = 0
        index = self._get_candidate_workflow_index()
        while index is not None:
            candidate_node = self._candidate_workflow_children_nodes(
                candidate_workflow=self.candidate_workflows[index])
            print("*******the " + str(execution) + "th search*******")
            print("\tselected workflow: " + str(self.candidate_workflows[index]))
            print("\tcandidate_node = " + str(candidate_node))
            self._ST_node_action(node=candidate_node[0])
            self._update_workflow_node(node=candidate_node[0])
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
            children_node = self._candidate_workflow_children_nodes(candidate_workflow=self.candidate_workflows[0])
            while children_node:
                self._ST_node_selected_workflow(node=children_node[0], candidate_workflow=self.candidate_workflows[0])
                self.candidate_workflows[0].node = children_node[0]
                children_node = self._candidate_workflow_children_nodes(candidate_workflow=self.candidate_workflows[0])
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
