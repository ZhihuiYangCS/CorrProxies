import sys
import time
from abc import ABC, abstractmethod
from typing import List, Tuple, Union
import rootpath

rootpath.append()
from utility.constant import UCF101_TYPE
from optimizer.accuracy_allocator.acc_allocator.accuracy_allocator_utility import AccCombPPFiltersMap, \
    AccCombPPFiltersList, generate_exhausted_search_space
from optimizer.PPs.pp_filter import PPFilter
from optimizer.accuracy_allocator.pp_manager_base import PPManagerBase
from optimizer.reorder.reorder_utility import WorkflowCandidate
from records.record import Record
from utility.utility import get_selectivity


def copy_labeled_samples(labeled_samples: List[List[Record]]) -> List[List[Record]]:
    """
    copy the labeled_samples
    """
    result_labeled_samples = []
    for labeled_sample in labeled_samples:
        result_labeled_samples.append(labeled_sample.copy())
    return result_labeled_samples


class AccuracyAllocatorReorderBase(ABC):
    """
    a base class for accuracy allocation in reorder
    """

    def __init__(self, workflow, eta: float):
        self.workflow = workflow
        self.eta = eta
        self.train_time = 0

    @abstractmethod
    def allocate_accuracy(self, labeled_samples: List[List[Record]], pp_managers: List[PPManagerBase],
                          candidate_workflow: WorkflowCandidate, final_validation_flag: bool = True,
                          first_node_input: Union[List[Record], None] = None):
        """
        perform accuracy allocation for the candidate_workflow
        :param first_node_input: all input for this candidate workflow
        :param target_accuracy: the target_accuracy
        :param labeled_samples: a list of labeled_samples,
                                the number of labeled_samples may be larger than that of pp_managers
        :param pp_managers: a list of pp_managers on t_nodes
        :param candidate_workflow: a candidate_workflow to perform accuracy allocation
        :param final_validation_flag: a flag to indicate whether to perform final_validation step or not
        :return: a list of final pp_filters
        """
        pass

    def subsequent_pp_selectivity_update(self, start_index: int, labeled_samples: List[List[Record]],
                                         pp_filters: List[PPFilter], candidate_workflow: WorkflowCandidate):
        """
        The method is useful when the number of s_nodes are more than that of t_nodes.
        The method are used to update subsequent s_nodes' pp_selectivities under the condition of trained pp_filters
        :param target_accuracy: the target_accuracy
        :param start_index: a int number indicate where to update pp_selectivity
        :param labeled_samples: all labeled_samples
        :param pp_filters: trained pp_filters
        :param candidate_workflow: the candidate_workflow to perform accuracy allocation
        """
        for i in range(start_index, len(labeled_samples)):
            labeled_sample = labeled_samples[i]
            for pp_filter in pp_filters:
                if self.workflow.workflow_type == UCF101_TYPE and self.workflow.opt_thread_num == 0:
                    pp_batch_output = pp_filter.process_batch(batch=labeled_sample, model=self.workflow.models[
                        self.workflow.preprocessor.operator_name])
                else:
                    pp_batch_output = pp_filter.multiple_threads_batch(batch=labeled_sample,
                                                                       num_process=self.workflow.opt_thread_num)
                labeled_sample = pp_batch_output.returned_batch
            max_pp_selectivity = min_pp_selectivity = get_selectivity(labeled_sample)
            no_t_count = i - start_index
            for j in range(no_t_count):
                min_pp_selectivity = min_pp_selectivity / self.workflow.target_accuracy - (
                        1 - self.workflow.target_accuracy) / self.workflow.target_accuracy
            candidate_workflow.min_pp_selectivities[i] = min_pp_selectivity
            candidate_workflow.max_pp_selectivities[i] = max_pp_selectivity

    def one_accuracy_combination_cost(self, accuracy_combination: List[float], labeled_samples: List[List[Record]],
                                      pp_managers: List[PPManagerBase], candidate_workflow: WorkflowCandidate,
                                      first_node_input: Union[List[Record], None] = None) -> AccCombPPFiltersMap:
        """
        one accuracy_combination search iteration
        :return: cost:float -> the cost of the workflow under the accuracy_combination condition
                 accuracy_combination:List[float] -> the search accuracy combination
                 pp_filter_list:List[PPFilter] -> a list of PPFilter trained under the accuracy_combination condition
                 candidate_workflow_copy: WorkflowCandidate -> a copy of candidate_workflow,
                                                               and updated under the accuracy_combination condition
                 flag: bool -> indicate the list of PPFilters are obtained from retrain or not.
                               True: from retrain; False: not from retrain
        """
        labeled_samples_copy = copy_labeled_samples(labeled_samples=labeled_samples)
        pp_filter_list, pp_retrained_flag = [], []
        candidate_workflow_copy = candidate_workflow.candidate_workflow_copy()
        for i in range(len(accuracy_combination)):
            candidate_workflow_copy.min_accuracy[i] = accuracy_combination[i]
            candidate_workflow_copy.max_accuracy[i] = accuracy_combination[i]
            labeled_sample = labeled_samples_copy[i]
            if i == 0 and pp_managers[i].pp_history:
                accuracy_info_dict = pp_managers[i].pp_history[0].accuracy_infos.get_accuracy_info(
                    accuracy=accuracy_combination[i])
                candidate_workflow_copy.min_reductions[i] = accuracy_info_dict.reduction
                candidate_workflow_copy.max_reductions[i] = accuracy_info_dict.reduction
                pp_filter_list.append(PPFilter(sub_plan=[i], estimator=pp_managers[i].pp_history[0].estimator,
                                               accuracy_info_dict=accuracy_info_dict,
                                               preprocessor=pp_managers[i].workflow.preprocessor))
                print("\t" + str(accuracy_info_dict))
            else:
                for j in range(0, i):
                    if self.workflow.workflow_type == UCF101_TYPE and self.workflow.opt_thread_num == 0:
                        pp_batch_output = pp_filter_list[j].process_batch(batch=labeled_sample,
                                                                          model=self.workflow.models[
                                                                              self.workflow.preprocessor.operator_name])
                    else:
                        pp_batch_output = pp_filter_list[j].multiple_threads_batch(batch=labeled_sample,
                                                                                   num_process=self.workflow.opt_thread_num)
                    labeled_sample = pp_batch_output.returned_batch
                pp_selectivity = get_selectivity(labeled_sample)
                candidate_workflow_copy.min_pp_selectivities[i] = pp_selectivity
                candidate_workflow_copy.max_pp_selectivities[i] = pp_selectivity
                print("\tsub_plan_sample num = " + str(len(labeled_sample)))
                X_train_feature, train_cache, X_validate_feature, validate_cache, X_test_feature, test_cache = \
                    pp_managers[i].data_set_convert(data_set=labeled_sample)
                time1 = time.time()
                prob_predictor, accuracy_info_dict, flag = pp_managers[i].get_pp_and_accuracy_info(
                    X_train_feature=X_train_feature, train_cache=train_cache,
                    X_validate_feature=X_validate_feature, validate_cache=validate_cache,
                    X_test_feature=X_test_feature, test_cache=test_cache, accuracy=accuracy_combination[i])
                time2 = time.time()
                self.train_time += (time2 - time1)
                candidate_workflow_copy.min_reductions[i] = accuracy_info_dict.reduction
                candidate_workflow_copy.max_reductions[i] = accuracy_info_dict.reduction
                pp_retrained_flag.append(flag)
                pp_filter_list.append(PPFilter(sub_plan=[i], estimator=prob_predictor.estimator,
                                               accuracy_info_dict=accuracy_info_dict,
                                               preprocessor=pp_managers[i].workflow.preprocessor))
                print("\t" + str(accuracy_info_dict))
        self.subsequent_pp_selectivity_update(start_index=len(pp_managers), labeled_samples=labeled_samples_copy,
                                              pp_filters=pp_filter_list, candidate_workflow=candidate_workflow_copy)
        if len(pp_managers) == candidate_workflow_copy.ml_filters_num:
            candidate_workflow_copy.min_cost = candidate_workflow_copy.get_cost_cc(
                samples=first_node_input, pp_filters=pp_filter_list, ml_filters=candidate_workflow_copy.ml_filters)
            candidate_workflow_copy.max_cost = candidate_workflow_copy.min_cost
        else:
            candidate_workflow_copy.min_cost = candidate_workflow_copy.get_min_cost()
            candidate_workflow_copy.max_cost = candidate_workflow_copy.get_max_cost()
        cost = (candidate_workflow_copy.min_cost + candidate_workflow_copy.max_cost) / 2
        acc_comb_pps = AccCombPPFiltersMap(accuracy_combination=accuracy_combination, cost=cost,
                                           pp_filters=pp_filter_list, retrain_flag=all(pp_retrained_flag),
                                           candidate_workflow=candidate_workflow_copy)
        print("\t" + str(acc_comb_pps))
        return acc_comb_pps

    def retrain_final_pps(self, accuracy_combination: List[float], first_pp: PPFilter,
                          labeled_samples: List[List[Record]], pp_managers: List[PPManagerBase],
                          candidate_workflow: WorkflowCandidate,
                          first_node_input: Union[List[Record], None] = None) -> Tuple[List[PPFilter], float]:
        """
        finally retrain PPs on the searched accuracy_combination with minimal cost;
        :param first_node_input: all input for this candidate workflow
        :param target_accuracy: the target_accuracy
        :param accuracy_combination: the searched accuracy_combination with minimal cost
        :param first_pp: the first pp which does not need to retrain
        :param labeled_samples: the labeled_samples for all s_nodes
        :param pp_managers: pp_managers for all t_nodes
        :param candidate_workflow: the candidate_workflow to perform accuracy allocation
        :return: finally retrained pp_filters
        """
        pp_filters = [first_pp]
        print("\t" + str(pp_filters[0].accuracy_info_dict))
        labeled_samples_copy = copy_labeled_samples(labeled_samples=labeled_samples)
        for i in range(1, len(accuracy_combination)):
            labeled_sample = labeled_samples_copy[i]
            for j in range(0, i):
                if self.workflow.workflow_type == UCF101_TYPE and self.workflow.opt_thread_num == 0:
                    pp_batch_output = pp_filters[j].process_batch(batch=labeled_sample, model=self.workflow.models[
                        self.workflow.preprocessor.operator_name])
                else:
                    pp_batch_output = pp_filters[j].multiple_threads_batch(batch=labeled_sample,
                                                                           num_process=self.workflow.opt_thread_num)
                labeled_sample = pp_batch_output.returned_batch
            pp_selectivity = get_selectivity(labeled_sample)
            candidate_workflow.min_pp_selectivities[i] = pp_selectivity
            candidate_workflow.max_pp_selectivities[i] = pp_selectivity
            X_train_feature, train_cache, X_validate_feature, validate_cache, X_test_feature, test_cache = \
                pp_managers[i].data_set_convert(data_set=labeled_sample)
            time1 = time.time()
            prob_predictor = pp_managers[i].retrain_pp(X_train_feature=X_train_feature, train_cache=train_cache,
                                                       X_validate_feature=X_validate_feature,
                                                       validate_cache=validate_cache, X_test_feature=X_test_feature,
                                                       test_cache=test_cache)
            time2 = time.time()
            self.train_time += (time2 - time1)
            accuracy_info_dict = prob_predictor.accuracy_infos.get_accuracy_info(
                accuracy=accuracy_combination[i])
            print("\t" + str(accuracy_info_dict))
            candidate_workflow.min_reductions[i] = accuracy_info_dict.reduction
            candidate_workflow.max_reductions[i] = accuracy_info_dict.reduction
            pp_filters.append(
                PPFilter(sub_plan=[i], estimator=prob_predictor.estimator, accuracy_info_dict=accuracy_info_dict,
                         preprocessor=pp_managers[i].workflow.preprocessor))
        self.subsequent_pp_selectivity_update(start_index=len(accuracy_combination),
                                              labeled_samples=labeled_samples_copy, pp_filters=pp_filters,
                                              candidate_workflow=candidate_workflow)
        if len(pp_managers) == candidate_workflow.ml_filters_num:
            candidate_workflow.min_cost = candidate_workflow.get_cost_cc(samples=first_node_input,
                                                                         pp_filters=pp_filters,
                                                                         ml_filters=candidate_workflow.ml_filters)
            candidate_workflow.max_cost = candidate_workflow.min_cost
        else:
            candidate_workflow.min_cost = candidate_workflow.get_min_cost()
            candidate_workflow.max_cost = candidate_workflow.get_max_cost()
        return pp_filters, (candidate_workflow.min_cost + candidate_workflow.max_cost) / 2


class AccuracyAllocatorExhaustedSearch(AccuracyAllocatorReorderBase):
    """
    the exhausted search method to allocate accuracy
    """

    def __init__(self, workflow, eta: float):
        super().__init__(workflow=workflow, eta=eta)

    def allocate_accuracy(self, labeled_samples: List[List[Record]], pp_managers: List[PPManagerBase],
                          candidate_workflow: WorkflowCandidate, final_validation_flag: bool = True,
                          first_node_input: Union[List[Record], None] = None):
        """
        perform accuracy allocation for the candidate_workflow
        :param first_node_input: all input for this candidate workflow
        :param target_accuracy: the target_accuracy
        :param labeled_samples: a list of labeled_samples,
                                the number of labeled_samples may be larger than that of pp_managers
        :param pp_managers: a list of pp_managers on t_nodes
        :param candidate_workflow: a candidate_workflow to perform accuracy allocation
        :param final_validation_flag: a flag to indicate whether to perform final_validation step or not
        :return: a list of final pp_filters
        """
        search_space = generate_exhausted_search_space(accuracy_num=len(pp_managers),
                                                       target_accuracy=self.workflow.target_accuracy, eta=self.eta)
        result_acc_comb_pps = AccCombPPFiltersMap(accuracy_combination=[], cost=sys.maxsize, pp_filters=[],
                                                  retrain_flag=True)
        search_index = 0
        for accuracy_combination in search_space:
            print("the " + str(search_index) + "th search step, accuracy = " + str(accuracy_combination))
            acc_comb_pps = self.one_accuracy_combination_cost(accuracy_combination=accuracy_combination,
                                                              labeled_samples=labeled_samples,
                                                              pp_managers=pp_managers,
                                                              candidate_workflow=candidate_workflow,
                                                              first_node_input=first_node_input)
            if acc_comb_pps.cost < result_acc_comb_pps.cost:
                result_acc_comb_pps = acc_comb_pps
            search_index += 1
        print("final_accuracy_combination = " + str(result_acc_comb_pps.accuracy_combination) + "\tfinal_cost = " + str(
            result_acc_comb_pps.cost) + "\tfrom retrain or not: " + str(result_acc_comb_pps.retrain_flag))
        print("result_candidate_workflow")
        result_acc_comb_pps.candidate_workflow.detail_info()
        candidate_workflow.candidate_workflow_reset(candidate_workflow=result_acc_comb_pps.candidate_workflow)
        if final_validation_flag and not result_acc_comb_pps.retrain_flag:
            print("now retrain")
            final_pps, result_cost = self.retrain_final_pps(
                accuracy_combination=result_acc_comb_pps.accuracy_combination,
                first_pp=result_acc_comb_pps.pp_filters[0], labeled_samples=labeled_samples,
                pp_managers=pp_managers, candidate_workflow=candidate_workflow, first_node_input=first_node_input)
        else:
            print("no need to retrain")
            final_pps = result_acc_comb_pps.pp_filters
            result_cost = result_acc_comb_pps.cost
        return final_pps, result_cost, self.train_time


class AccuracyAllocatorHillClimbing(AccuracyAllocatorReorderBase):
    """
    the hill climbing method to allocate accuracy
    """

    def __init__(self, workflow, eta: float):
        super().__init__(workflow=workflow, eta=eta)

    def _one_step_climbing(self, i: int, j: int, accuracy_combination: List[float], labeled_samples: List[List[Record]],
                           pp_managers: List[PPManagerBase], candidate_workflow: WorkflowCandidate,
                           search_acc_comb_pp_list: AccCombPPFiltersList,
                           first_node_input: Union[List[Record], None] = None) -> Union[AccCombPPFiltersMap, None]:
        accuracy_combination_copy = accuracy_combination.copy()
        accuracy_combination_copy[i] *= self.eta
        accuracy_combination_copy[j] /= self.eta
        if self.workflow.target_accuracy <= accuracy_combination_copy[i] <= 1.0 and \
                self.workflow.target_accuracy <= accuracy_combination_copy[j] <= 1.0:
            acc_comb_pp = search_acc_comb_pp_list.search(accuracy_combination=accuracy_combination_copy)
            if acc_comb_pp is None:
                acc_comb_pp = self.one_accuracy_combination_cost(accuracy_combination=accuracy_combination_copy,
                                                                 labeled_samples=labeled_samples,
                                                                 pp_managers=pp_managers,
                                                                 candidate_workflow=candidate_workflow,
                                                                 first_node_input=first_node_input)
                search_acc_comb_pp_list.add(acc_pp=acc_comb_pp)
                return acc_comb_pp
        return None

    def _allocate_accuracy_one_hill_climbing(self, accuracy_combination: List[float],
                                             labeled_samples: List[List[Record]],
                                             pp_managers: List[PPManagerBase], candidate_workflow: WorkflowCandidate,
                                             search_acc_comb_pp_list: AccCombPPFiltersList,
                                             first_node_input: Union[List[Record], None] = None):
        acc_pps_list = AccCombPPFiltersList()
        for i in range(len(accuracy_combination)):
            for j in range(i + 1, len(accuracy_combination)):
                print(str(i) + "," + str(j))
                acc_comb_pp_1 = self._one_step_climbing(i=i, j=j, accuracy_combination=accuracy_combination,
                                                        labeled_samples=labeled_samples, pp_managers=pp_managers,
                                                        candidate_workflow=candidate_workflow,
                                                        search_acc_comb_pp_list=search_acc_comb_pp_list,
                                                        first_node_input=first_node_input)
                if acc_comb_pp_1 is not None:
                    acc_pps_list.add(acc_pp=acc_comb_pp_1)
                acc_comb_pp_2 = self._one_step_climbing(i=j, j=i, accuracy_combination=accuracy_combination,
                                                        labeled_samples=labeled_samples, pp_managers=pp_managers,
                                                        candidate_workflow=candidate_workflow,
                                                        search_acc_comb_pp_list=search_acc_comb_pp_list,
                                                        first_node_input=first_node_input)
                if acc_comb_pp_2 is not None:
                    acc_pps_list.add(acc_pp=acc_comb_pp_2)
        return acc_pps_list.min()

    def _allocate_accuracy_hill_climbing(self, accuracy_combination: List[float], labeled_samples: List[List[Record]],
                                         pp_managers: List[PPManagerBase], candidate_workflow: WorkflowCandidate,
                                         search_acc_comb_pp_list: AccCombPPFiltersList,
                                         first_node_input: Union[List[Record], None] = None):
        old_acc_comb_pps = search_acc_comb_pp_list.search(accuracy_combination=accuracy_combination)
        if old_acc_comb_pps is None:
            old_acc_comb_pps = self.one_accuracy_combination_cost(accuracy_combination=accuracy_combination,
                                                                  labeled_samples=labeled_samples,
                                                                  pp_managers=pp_managers,
                                                                  candidate_workflow=candidate_workflow,
                                                                  first_node_input=first_node_input)
            search_acc_comb_pp_list.add(acc_pp=old_acc_comb_pps)

        new_acc_comb_pps = self._allocate_accuracy_one_hill_climbing(accuracy_combination=accuracy_combination,
                                                                     labeled_samples=labeled_samples,
                                                                     pp_managers=pp_managers,
                                                                     candidate_workflow=candidate_workflow,
                                                                     search_acc_comb_pp_list=search_acc_comb_pp_list,
                                                                     first_node_input=first_node_input)
        count = 0
        print("the " + str(count) + " step, old " + str(old_acc_comb_pps) + "\nnew = " + str(new_acc_comb_pps))
        while new_acc_comb_pps is not None and new_acc_comb_pps.cost < old_acc_comb_pps.cost * 1.01:
            old_acc_comb_pps = new_acc_comb_pps
            new_acc_comb_pps = self._allocate_accuracy_one_hill_climbing(
                accuracy_combination=old_acc_comb_pps.accuracy_combination, labeled_samples=labeled_samples,
                pp_managers=pp_managers, candidate_workflow=candidate_workflow,
                search_acc_comb_pp_list=search_acc_comb_pp_list, first_node_input=first_node_input)
            count += 1
            print("the " + str(count) + " step, old " + str(old_acc_comb_pps) + "\nnew = " + str(new_acc_comb_pps))

    def allocate_accuracy(self, labeled_samples: List[List[Record]], pp_managers: List[PPManagerBase],
                          candidate_workflow: WorkflowCandidate, final_validation_flag: bool = True,
                          first_node_input: Union[List[Record], None] = None):
        """
        perform accuracy allocation for the candidate_workflow
        :param first_node_input: all input for this candidate workflow
        :param target_accuracy: the target_accuracy
        :param labeled_samples: a list of labeled_samples,
                                the number of labeled_samples may be larger than that of pp_managers
        :param pp_managers: a list of pp_managers on t_nodes
        :param candidate_workflow: a candidate_workflow to perform accuracy allocation
        :param final_validation_flag: a flag to indicate whether to perform final_validation step or not
        :return: a list of final pp_filters
        """
        search_acc_comb_pp_list = AccCombPPFiltersList()
        initial_accuracy = [self.workflow.target_accuracy] + [1.00] * (len(pp_managers) - 1)
        self._allocate_accuracy_hill_climbing(accuracy_combination=initial_accuracy, labeled_samples=labeled_samples,
                                              pp_managers=pp_managers, candidate_workflow=candidate_workflow,
                                              search_acc_comb_pp_list=search_acc_comb_pp_list,
                                              first_node_input=first_node_input)
        # one jump, search from another start point
        # if len(initial_accuracy) > 2:
        #     initial_accuracy = [1.00] * (len(pp_managers) - 1) + [self.workflow.target_accuracy]
        #     self._allocate_accuracy_hill_climbing(accuracy_combination=initial_accuracy,
        #                                           labeled_samples=labeled_samples,
        #                                           pp_managers=pp_managers, candidate_workflow=candidate_workflow,
        #                                           search_acc_comb_pp_list=search_acc_comb_pp_list,
        #                                           first_node_input=first_node_input)
        acc_comb_pp_filters = search_acc_comb_pp_list.min()
        print("result_candidate_workflow")
        acc_comb_pp_filters.candidate_workflow.detail_info()
        candidate_workflow.candidate_workflow_reset(candidate_workflow=acc_comb_pp_filters.candidate_workflow)
        if final_validation_flag and not acc_comb_pp_filters.retrain_flag:
            print("now retrain")
            final_pps, result_cost = self.retrain_final_pps(
                accuracy_combination=acc_comb_pp_filters.accuracy_combination,
                first_pp=acc_comb_pp_filters.pp_filters[0], labeled_samples=labeled_samples,
                pp_managers=pp_managers, candidate_workflow=candidate_workflow, first_node_input=first_node_input)
        else:
            print("no need to retrain")
            final_pps = acc_comb_pp_filters.pp_filters
            result_cost = acc_comb_pp_filters.cost
        return final_pps, result_cost, self.train_time
