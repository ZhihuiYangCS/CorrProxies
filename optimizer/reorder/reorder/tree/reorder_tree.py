import time
from abc import ABC, abstractmethod
from typing import List, Union

import numpy
import rootpath

rootpath.append()
from utility.constant import TWITTER_TYPE, UCF101_TYPE
from utility.utility_base import copy_filter_batch_out_result_all
from optimizer.PPs.pp_filter import PPFilter
from operators.ml_filter import MLFilter
from records.record import Record
from utility.utility import batch_records_delete_ml_operator_results, sample_train_stop_condition, batch_records_copy
from optimizer.reorder.reorder.reorder_base import ReorderBase
from optimizer.reorder.reorder_utility import MyNode, WorkflowCandidate, print_workflow_candidates


class ReorderTree(ReorderBase, ABC):
    """
    a base class for tree-based methods
        _generate_S_node_labeled_samples:
            _generate_S_node_labeled_samples: support generate S_node labeled_samples one record by one record
            _generate_S_node_block_labeled_samples: support generate s_node labeled_samples one block by one block
        search utilities:
            _exist_search method
            _candidate_workflow_children_nodes
            _update_workflow_node
            _update_workflow_cost
    """

    def __init__(self, workflow):
        super().__init__(workflow=workflow)
        self.scan_root = None

    def initialization(self):
        """
        1. generate candidate_workflows
        2. generate search tree.
        """
        self.scan_root = MyNode(node_id=0, node_type="R", operator=self.scan)
        time1 = time.time()
        self.generate_candidate_workflows(scan_root=self.scan_root)
        print("generate candidate workflows time = " + str(time.time() - time1))
        print_workflow_candidates(self.candidate_workflows)
        self.generate_search_tree()

    @abstractmethod
    def generate_search_tree(self):
        """
        generate a search tree, enumerate all possible plan tree or fine grained S-T search tree
        """
        pass

    def _generate_first_t_node(self, node: MyNode, operator_index: int):
        """
        generate first t_node child for the parent node
        :param node: parent node
        :param operator_index: int
        :return: operator_index: int, nodes:List[MyNode]
        """
        nodes = []
        s_nodes, t_nodes = node.get_pass_st_nodes()
        # including node itselt
        if node.node_type == "S":
            s_nodes.append(node)
        if node.node_type == "T":
            t_nodes.append(node)
        if len(s_nodes) != len(t_nodes):
            nodes.append(
                MyNode(node_id=operator_index, node_type="T", operator=s_nodes[len(t_nodes)].operator, parent=node))
            operator_index += 1
        return operator_index, nodes

    def _check_insert_s_nodes(self, s_nodes: List[MyNode], ml_filter: MLFilter):
        """
        check if we can insert this ml_filter;
        :param s_nodes:
        :param ml_filter:
        :return:
        """
        flag = False
        nodes_ml_filters_operator_names = []
        for node in s_nodes:
            nodes_ml_filters_operator_names.append(node.operator.operator_name)
        nodes_ml_filters_operator_names.append(ml_filter.operator_name)
        for candidate_workflow in self.candidate_workflows:
            ml_filter_names = candidate_workflow.workflow.get_ml_filters_operator_names()
            candidate_workflow_flag = True
            for index, operator_name in enumerate(nodes_ml_filters_operator_names):
                if operator_name != ml_filter_names[index]:
                    candidate_workflow_flag = False
                    break
            if candidate_workflow_flag:
                flag = candidate_workflow_flag
                break
        return flag

    def _generate_all_s_nodes(self, node: MyNode, operator_index: int):
        """
        generate all possible s_nodes children for the parent node
        :param node: parent node
        :return: nodes:List[MyNode]
        """
        nodes = []
        s_nodes, t_nodes = node.get_pass_st_nodes()
        # including node itselt
        if node.node_type == "S":
            s_nodes.append(node)
        if node.node_type == "T":
            t_nodes.append(node)
        s_nodes_filter_names = []
        s_nodes_ml_filters = []
        for s_node in s_nodes:
            s_nodes_filter_names.append(s_node.operator.operator_name)
            s_nodes_ml_filters.append(s_node.operator)
        remaining_ml_filters = list(numpy.array(self.ml_filters)[
                                        numpy.invert(numpy.in1d(self.ml_filters_operator_names, s_nodes_filter_names))])
        for ml_filter in remaining_ml_filters:
            if self._check_insert_s_nodes(s_nodes=s_nodes, ml_filter=ml_filter):
                nodes.append(MyNode(node_id=operator_index, node_type="S",
                                    operator=ml_filter.get_new_object(previous_operators=s_nodes_ml_filters,
                                                                      operator_name=ml_filter.operator_name.split("_")[
                                                                          -1]), parent=node))
                operator_index += 1
        return operator_index, nodes

    def _pass_nodes_equals_candidate_workflow(self, candidate_workflow: WorkflowCandidate, nodes: List[MyNode]):
        """
        check whether pass nodes equals to the candidate_workflow or not.
        :param candidate_workflow: a WorkflowCandidate
        :param nodes: a list of pass nodes
        :return: flag: bool; True -> Equal; False -> Not equal
        """
        flag = True
        ml_filters = candidate_workflow.workflow.get_ml_filters()
        for i in range(len(nodes)):
            if nodes[i].operator.operator_name != ml_filters[i].operator_name:
                flag = False
        return flag

    def _candidate_workflow_children_nodes(self, candidate_workflow: WorkflowCandidate):
        """
        get the candidate_workflow's all children nodes.
        :return:candidate_nodes:List[MyNode]
        """
        candidate_nodes = []
        ml_filters = candidate_workflow.workflow.get_ml_filters()
        children_nodes = candidate_workflow.node.children
        for child_node in children_nodes:
            if child_node.node_type == "T":
                candidate_nodes.append(child_node)
            if child_node.node_type == "S":
                s_nodes, t_nodes = child_node.get_pass_st_nodes()
                if ml_filters[len(s_nodes)].operator_name == child_node.operator.operator_name:
                    candidate_nodes.append(child_node)
        return candidate_nodes

    def _get_node_match_candidate_workflows(self, node: MyNode) -> List[WorkflowCandidate]:
        """
        get all match candidate workflows for this node
        :param node: a node
        :return: a list of matching candidate_workflows
        """
        match_candidate_workflows = []
        pass_nodes = node.get_pass_workflow_nodes()
        for candidate_workflow in self.candidate_workflows:
            if candidate_workflow is not None and self._pass_nodes_equals_candidate_workflow(
                    candidate_workflow=candidate_workflow, nodes=pass_nodes):
                match_candidate_workflows.append(candidate_workflow)
        return match_candidate_workflows

    def _update_workflow_node(self, node: MyNode):
        """
        when search one node and perform the node's corresponding action, update all candidate workflows node.
        :param node: a searched node
        """
        match_candidate_workflows = self._get_node_match_candidate_workflows(node=node)
        if node.node_type == "S":
            for candidate_workflow in match_candidate_workflows:
                candidate_workflow.node = node
        if node.node_type == "T":
            for candidate_workflow in match_candidate_workflows:
                children_nodes = candidate_workflow.node.children
                for child_node in children_nodes:
                    if child_node.node_type == "T":
                        candidate_workflow.node = child_node

    def _update_workflow_cost(self):
        """
        update all modified candidate workflows costs
        """
        if self.constant_cost.flag:
            for candidate_workflow in self.candidate_workflows:
                candidate_workflow.detail_info()
                candidate_workflow.min_cost = candidate_workflow.get_min_cost()
                candidate_workflow.max_cost = candidate_workflow.get_max_cost()
                candidate_workflow.flag = False
            self.constant_cost.flag = False
        else:
            for candidate_workflow in self.candidate_workflows:
                candidate_workflow.detail_info()
                if candidate_workflow.flag:
                    candidate_workflow.min_cost = candidate_workflow.get_min_cost()
                    candidate_workflow.max_cost = candidate_workflow.get_max_cost()
                    candidate_workflow.flag = False

    def _go_back_get_cache_nodes(self, node) -> List[MyNode]:
        """
        go back to get the first node which has cache or is Root node
        :return: nodes:List[MyNode]
        """
        nodes = []
        path_nodes = node.get_all_pass_nodes()
        for i in range(len(path_nodes) - 1, -1, -1):
            if path_nodes[i].cache or path_nodes[i].node_type == "R":
                nodes = path_nodes[i:]
                break
        return nodes

    def _s_node_process_one_record(self, record: Record, s_node: MyNode, s_node_names_flags: dict) -> Union[
        Record, None]:
        """
        this method implements one s_node.operator process one record;
        if this s_node is the first s_node: append this record to its first_s_node_input attribute;
        if this s_node's sample_train_stop_condition doesn't satisfy, make a copy of this record, label it and add it to this node's labeled_samples
        process this record
        if we need to delete this record's ml_operator's result: delete it.
        """
        s_nodes, t_nodes = s_node.get_pass_st_nodes()
        if len(s_nodes) == 0:
            s_node.first_s_node_input.append(record)
        return_record = None
        if record is not None:
            if not sample_train_stop_condition(workflow_type=self.workflow.workflow_type,
                                               process_num=s_node.process_num,
                                               pos_num=s_node.pass_num):
                labeled_record = record.copy()
            else:
                labeled_record = None
            batch_output = s_node.operator.process_one(record=record, model=self.workflow.models[
                s_node.operator.ml_predicator.ml_operator.operator_name])
            if batch_output.returned_batch:
                return_record = batch_output.returned_batch[0]
            else:
                return_record = None
            if s_node_names_flags[s_node.operator.operator_name]:
                batch_records_delete_ml_operator_results(batch_records=[record],
                                                         ml_operators=[s_node.operator.ml_predicator.ml_operator])
            if labeled_record is not None:
                s_node.process_num += 1
                if s_node.operator.ml_predicator.get_result(record):
                    labeled_record["flag"] = 1
                    s_node.pass_num += 1
                else:
                    labeled_record["flag"] = 0
                s_node.labeled_samples.append(labeled_record)
        return return_record

    def _get_one_record(self, node: MyNode) -> Union[Record, None]:
        """
        get one record from the node where the node.node_type is "R", "S" or "T"
        :param node:
        :return: a record: Record
        """
        if node.node_type == "R":
            record = node.operator.get_one_record()
        else:
            if node.cache:
                record = node.cache.pop(0)
            else:
                record = None
        return record

    def _one_record_go_through_path(self, record: Record, path_nodes: List[MyNode], s_node_names_flags: dict):
        """
        one record go through the path nodes
        if the first node in the path is "R" root-node: preprocess this record;
        if the first node in the path is "S" S_node: call _s_node_process_one_record function to process this record
        pass this record to sibling_nodes
        pass the output record to children_node.
        """
        if path_nodes[0].node_type == "R" and record is not None:
            batch_output = self.workflow.preprocessor.process_one(record=record, model=self.workflow.models[
                self.workflow.preprocessor.operator_name])
            record = batch_output.returned_batch[0]
        if path_nodes[0].node_type == "S" and record is not None:
            record = self._s_node_process_one_record(record=record, s_node=path_nodes[0],
                                                     s_node_names_flags=s_node_names_flags)
        if len(path_nodes) > 1:
            for current_node in path_nodes[1:]:
                sibling_nodes = list(current_node.siblings)  # brother nodes
                for sibling_node in sibling_nodes:
                    if record is not None:
                        sibling_node.cache += [record]
                if current_node.node_type == "S":
                    record = self._s_node_process_one_record(record=record, s_node=current_node,
                                                             s_node_names_flags=s_node_names_flags)
        if path_nodes[-1].children:
            for current_node in path_nodes[-1].children:
                if record is not None:
                    current_node.cache += [record]
        return record

    def _check_s_node_delete_ml_operator_result(self, s_node: MyNode) -> bool:
        """
        check delete the record's ml_operator result;
        if s_node's children nodes contain s_node with the same ml_operator operator_name, not delete
        else: delete
        :param record: a record contains ml_operator result
        """
        children_nodes = s_node.children
        delete_flag = True
        if children_nodes:
            for child_node in children_nodes:
                if child_node.node_type == "S" and child_node.operator.ml_predicator.ml_operator.operator_name \
                        == s_node.operator.ml_predicator.ml_operator.operator_name:
                    delete_flag = False
        return delete_flag

    def _compute_s_nodes_process_delete_ml_operator_result_flag(self, node: MyNode) -> dict:
        """
        compute all path s_nodes, including itself, delete flag
        :param node: the node
        :return: a dict map s_node.operator.operator_name to a bool flag
        """
        path_s_nodes, path_t_nodes = node.get_pass_st_nodes()
        if node.node_type == "S":
            path_s_nodes.append(node)
        s_node_names_flags_dict = {}
        for s_node in path_s_nodes:
            s_node_names_flags_dict[s_node.operator.operator_name] = self._check_s_node_delete_ml_operator_result(
                s_node=s_node)
        return s_node_names_flags_dict

    def _generate_S_node_labeled_samples(self, node: MyNode, predicate=lambda a, b, c: a):
        """
        generate s_node labeled samples one record by one record
        """
        count = 0
        s_node_names_flags = self._compute_s_nodes_process_delete_ml_operator_result_flag(node=node)
        while not bool(predicate(self.workflow.workflow_type, node.process_num, node.pass_num)):
            path_nodes = self._go_back_get_cache_nodes(node=node)
            if count % 1000 == 0:
                print("\tpath_nodes = " + str(path_nodes) + "\tcount = " + str(count))
            record = self._get_one_record(node=path_nodes[0])
            if path_nodes[0] == "R" and record is None:
                break
            self._one_record_go_through_path(record=record, path_nodes=path_nodes,
                                             s_node_names_flags=s_node_names_flags)
            count += 1
            if count % 1000 == 0:
                print("\tpath_nodes = " + str(path_nodes) + "\tcount = " + str(count))

    def _get_batch_record(self, node: MyNode) -> List[Record]:
        """
        get one batch from the node where the node.node_type is "R", "S" or "T"
        :param node:
        :return: a batch: List[Record]
        """
        if node.node_type == "R":
            batch_records = node.operator.get_block_records(self.workflow.batch_size)
        else:
            if node.cache:
                batch_records = node.cache[:self.workflow.batch_size]
                node.cache = node.cache[self.workflow.batch_size:]
            else:
                batch_records = []
        return batch_records

    def _batch_records_go_through_path(self, batch_records: List[Record], path_nodes: List[MyNode],
                                       s_node_names_flags: dict):
        """
        one batch go through the path nodes
        if the first node in the path is "R" root-node: preprocess this batch;
        if the first node in the path is "S" S_node: call _s_node_process_batch_records function to process this batch
        pass this batch to sibling_nodes
        pass the output batch to children_node.
        """
        if path_nodes[0].node_type == "R" and batch_records:
            if self.workflow.workflow_type == UCF101_TYPE and self.workflow.opt_thread_num == 0:
                batch_output = self.workflow.preprocessor.process_batch(batch=batch_records, model=self.workflow.models[
                    self.workflow.preprocessor.operator_name])
            else:
                batch_output = self.workflow.preprocessor.multiple_processes_batch(batch=batch_records,
                                                                                   num_process=self.workflow.opt_thread_num)
            batch_records = batch_output.returned_batch
        block_records = batch_records.copy()
        if path_nodes[0].node_type == "S" and batch_records:
            result_batch = self._s_node_process_batch_records(batch_records=batch_records, s_node=path_nodes[0],
                                                              s_node_names_flags=s_node_names_flags)
            copy_filter_batch_out_result_all(large_batch_records=block_records,
                                             small_batch_records_has_results=batch_records)
            batch_records = result_batch
        if len(path_nodes) > 1:
            for current_node in path_nodes[1:]:
                sibling_nodes = list(current_node.siblings)  # brother nodes
                for sibling_node in sibling_nodes:
                    if batch_records:
                        sibling_node.cache += batch_records
                if current_node.node_type == "S":
                    result_batch = self._s_node_process_batch_records(batch_records=batch_records, s_node=current_node,
                                                                      s_node_names_flags=s_node_names_flags)
                    copy_filter_batch_out_result_all(large_batch_records=block_records,
                                                     small_batch_records_has_results=batch_records)
                    batch_records = result_batch
        s_nodes, t_nodes = path_nodes[-1].get_pass_st_nodes()
        if path_nodes[-1].node_type == "S":
            s_nodes.append(path_nodes[-1])
        if path_nodes[0].node_type == "T":
            copy_filter_batch_out_result_all(large_batch_records=s_nodes[0].first_s_node_input,
                                             small_batch_records_has_results=block_records)
        else:
            s_nodes[0].first_s_node_input += block_records
        if path_nodes[-1].children:
            for current_node in path_nodes[-1].children:
                if batch_records:
                    current_node.cache += batch_records
        return batch_records

    def _s_node_process_batch_records(self, batch_records: List[Record], s_node: MyNode, s_node_names_flags: dict) -> \
            List[Record]:
        """
        this method implements one s_node.operator process one batch;
        if this s_node is the first s_node: append this batch to its first_s_node_input attribute;
        if this s_node's sample_train_stop_condition doesn't satisfy, make a copy of this batch, label it and add it to this node's labeled_samples
        multiple thread process this batch
        if we need to delete this record's ml_operator's result: delete it.
        """
        if not sample_train_stop_condition(workflow_type=self.workflow.workflow_type,
                                           process_num=s_node.process_num,
                                           pos_num=s_node.pass_num):
            labeled_samples = batch_records_copy(batch_records=batch_records)
        else:
            labeled_samples = None
        if self.workflow.workflow_type == UCF101_TYPE and self.workflow.opt_thread_num == 0:
            batch_output = s_node.operator.process_batch(batch=batch_records, model=self.workflow.models[
                s_node.operator.ml_predicator.ml_operator.operator_name])
        elif self.workflow.workflow_type == TWITTER_TYPE:
            batch_output = s_node.operator.multiple_threads_batch(batch=batch_records,
                                                                  num_process=self.workflow.opt_thread_num,
                                                                  model=self.workflow.models[
                                                                      s_node.operator.ml_predicator.ml_operator.operator_name])
        else:
            batch_output = s_node.operator.multiple_processes_batch(batch=batch_records,
                                                                    num_process=self.workflow.opt_thread_num)
        result_batch = batch_output.returned_batch
        if s_node_names_flags[s_node.operator.operator_name]:
            batch_records_delete_ml_operator_results(batch_records=batch_records,
                                                     ml_operators=[s_node.operator.ml_predicator.ml_operator])
        if labeled_samples is not None:
            for index, record in enumerate(labeled_samples):
                if not sample_train_stop_condition(workflow_type=self.workflow.workflow_type,
                                                   process_num=s_node.process_num,
                                                   pos_num=s_node.pass_num):
                    s_node.process_num += 1
                    if s_node.operator.ml_predicator.get_result(batch_records[index]):
                        s_node.pass_num += 1
                        record["flag"] = 1
                    else:
                        record["flag"] = 0
                    s_node.labeled_samples.append(record)
                else:
                    break
        return result_batch

    def _generate_S_node_labeled_samples_batch_process(self, node: MyNode, predicate=lambda a, b, c: a):
        count = 0
        s_node_names_flags = self._compute_s_nodes_process_delete_ml_operator_result_flag(node=node)
        while not bool(predicate(self.workflow.workflow_type, node.process_num, node.pass_num)):
            path_nodes = self._go_back_get_cache_nodes(node=node)
            print("\tpath_nodes = " + str(path_nodes) + "\tcount = " + str(count))
            batch_records = self._get_batch_record(node=path_nodes[0])
            if path_nodes[0] == "R" and not batch_records:
                break
            self._batch_records_go_through_path(batch_records=batch_records, path_nodes=path_nodes,
                                                s_node_names_flags=s_node_names_flags)
            count += len(batch_records)
            print("\tpath_nodes = " + str(path_nodes) + "\tcount = " + str(count))

    def _S_update_workflow_selectivity(self, match_workflows: List[WorkflowCandidate], index: int, selectivity: float,
                                       no_t_count: int, min_pp_selectivity: float, max_pp_selectivity: float):
        """
        updating matching candidate_workflows' min_selectivities, max_selectivities,
            min_pp_selectivities, max_pp_selectivities, min_reductions
        :param match_workflows: a list of matching candidate_workflows
        :param no_t_count: the number of unpassed t_nodes; we need to estimate pp_selectivity
        """
        for i in range(no_t_count):
            min_pp_selectivity = min_pp_selectivity / self.workflow.target_accuracy - (
                    1 - self.workflow.target_accuracy) / self.workflow.target_accuracy
            max_pp_selectivity = max_pp_selectivity / self.workflow.target_accuracy
        for workflow in match_workflows:
            workflow.flag = True
            workflow.min_selectivities[index] = selectivity
            workflow.max_selectivities[index] = selectivity
            workflow.min_pp_selectivities[index] = min_pp_selectivity
            workflow.max_pp_selectivities[index] = max_pp_selectivity
            workflow.min_reductions[index] = 1 - min_pp_selectivity * self.workflow.target_accuracy

    def _t_node_train_pp(self, t_node: MyNode, data_set: List[Record], accuracy: float) -> PPFilter:
        """
        train a pp_filter for the t_node
        :param t_node: the t_node to train pp_filter
        :param data_set: the labeled data_set
        :param accuracy: the desired accuracy
        :return: a trained pp_filter
        """
        X_train_feature, train_cache, X_validate_feature, validate_cache, X_test_feature, test_cache = \
            t_node.pp_manager.data_set_convert(data_set=data_set)
        proba_predicator, accuracy_info_dict, flag = t_node.pp_manager.get_pp_and_accuracy_info(
            X_train_feature=X_train_feature, train_cache=train_cache, X_validate_feature=X_validate_feature,
            validate_cache=validate_cache, X_test_feature=X_test_feature, test_cache=test_cache, accuracy=accuracy)
        pp_filter = PPFilter(sub_plan=[], estimator=proba_predicator.estimator,
                             accuracy_info_dict=accuracy_info_dict, preprocessor=self.workflow.preprocessor)
        return pp_filter

    def _get_candidate_workflow_index(self) -> Union[int, None]:
        """

        :return:
        """
        index = None
        if len(self.candidate_workflows) > 1:
            for i in range(len(self.candidate_workflows)):
                if self.candidate_workflows[i].node.children:
                    index = i
                    break
        return index

    @abstractmethod
    def _one_workflow_allocate_accuracy(self, candidate_workflow: WorkflowCandidate):
        """
        This method allocate accuracy for a candidate_workflow.
        The candidate_workflow should reach to leave node. The candidate_workflow.node should be a leave node.
        :param candidate_workflow: a candidate_workflow to allocate accuracy
        """
        pass
    # def reorder_optimization(self):
    #     """
    #     an unified interface for reorder optimization
    #     :return: pp_filters: a list of PPFilter
    #              ml_filters: a list of MLFilter
    #     """
    #     self.initialization()
    #     pp_filters, ml_filters, time_log = self.search()
    #     self.reorder_close()
    #     return pp_filters, ml_filters, time_log
