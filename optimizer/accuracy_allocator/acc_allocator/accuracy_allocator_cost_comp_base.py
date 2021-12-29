import time
from abc import ABC
from typing import List
import rootpath

rootpath.append()
from utility.constant import UCF101_TYPE
from operators.ml_filter import MLFilter
from optimizer.PPs.pp_filter import PPFilter
from optimizer.accuracy_allocator.acc_allocator.accuracy_allocator_base import AccuracyAllocatorBase
from optimizer.accuracy_allocator.acc_allocator.accuracy_allocator_utility import AccCombPPFiltersMap
from records.record import Record


class AccuracyAllocatorCostCompBase(AccuracyAllocatorBase, ABC):
    """
    allocation_mode: 0 ---- resample retrain
                     1 ---- reuse sample, retrain
                     2 ---- reuse sample, reuse PPs
    """

    def __init__(self, workflow, plan: List[List[int]], workflow_samples: List[Record] = None):
        super().__init__(workflow=workflow, plan=plan, workflow_samples=workflow_samples)

    def compute_cost(self, pp_filter_list: List[PPFilter], ml_filter_list: List[MLFilter]):
        """
        compute cost under an accuracy combination, which is computed by operator cost and filter selectivity
        compute final accuracy of the optimized workflow with PPs inserted.
        :param pp_filter_list: a list of PPs, which are inserted at the source
        :param ml_filter_list: a list of MLFilters in the workflow
        :return: workflow cost and accuracy
        """
        time1 = time.time()
        cost = 0
        all_samples = self.workflow_samples.copy()
        all_samples_num = len(all_samples)
        for pp_filter in pp_filter_list:
            cost += len(all_samples) * pp_filter.accuracy_info_dict.infer_cost
            if self.workflow.workflow_type == UCF101_TYPE and self.workflow.opt_thread_num == 0:
                pp_batch_output = pp_filter.process_batch(batch=all_samples, model=self.workflow.models[
                    self.workflow.preprocessor.operator_name])
            else:
                pp_batch_output = pp_filter.multiple_threads_batch(batch=all_samples,
                                                                   num_process=self.workflow.opt_thread_num)
            all_samples = pp_batch_output.returned_batch
        for ml_filter in ml_filter_list:
            cost += len(all_samples) * ml_filter.operator_cost
            if self.workflow.workflow_type == UCF101_TYPE and self.workflow.opt_thread_num == 0:
                filter_batch_output = ml_filter.process_batch(batch=all_samples, model=self.workflow.models[
                    ml_filter.ml_predicator.ml_operator.operator_name])
            else:
                filter_batch_output = ml_filter.multiple_threads_batch(batch=all_samples,
                                                                       num_process=self.workflow.opt_thread_num)
            all_samples = filter_batch_output.returned_batch
        time2 = time.time()
        print("compute cost time = " + str(time2 - time1))
        return cost / all_samples_num

    def one_accuracy_combination_cost(self, accuracy_combination: List[float]) -> AccCombPPFiltersMap:
        """
        compute the cost for one accuracy combination
        :param accuracy_combination: the accuracy combination that we compute the cost
        :return: an accuracy_combination_pp_filters_map
        """
        self._get_samples()
        for ml_filter in self.ml_filters:
            ml_filter.get_operator_cost()
            if ml_filter is None:
                ml_filter.print_operator_detail()
                exit(1)
        pp_filters_list, retrained_pps_accuracy = [], []
        for i in range(len(accuracy_combination)):
            sub_plan_sample = self.plan_sample_labeler.subplan_samples[i].copy()
            if i == 0 and self.plan_pp_managers[i].pp_history:
                accuracy_info_dict = self.plan_pp_managers[i].pp_history[0].accuracy_infos.get_accuracy_info(
                    accuracy=accuracy_combination[i])
                pp_filters_list.append(
                    PPFilter(sub_plan=self.plan[i], estimator=self.plan_pp_managers[i].pp_history[0].estimator,
                             accuracy_info_dict=accuracy_info_dict, preprocessor=self.workflow.preprocessor,
                             operator_name=str(self.plan) + "_" + str(i)))
                print("\t" + str(accuracy_info_dict))
            else:
                for j in range(0, i):
                    if self.workflow.workflow_type == UCF101_TYPE and self.workflow.opt_thread_num == 0:
                        pp_batch_output = pp_filters_list[j].process_batch(batch=sub_plan_sample,
                                                                           model=self.workflow.models[
                                                                               self.workflow.preprocessor.operator_name])
                    else:
                        pp_batch_output = pp_filters_list[j].multiple_threads_batch(batch=sub_plan_sample,
                                                                                    num_process=self.workflow.opt_thread_num)
                    sub_plan_sample = pp_batch_output.returned_batch
                print("\tsub_plan_sample num = " + str(len(sub_plan_sample)))
                X_train_feature, train_cache, X_validate_feature, validate_cache, X_test_feature, test_cache = \
                    self.plan_pp_managers[i].data_set_convert(sub_plan_sample)
                time1 = time.time()
                prob_predictor, accuracy_info_dict, flag = self.plan_pp_managers[i].get_pp_and_accuracy_info(
                    X_train_feature=X_train_feature, train_cache=train_cache, X_validate_feature=X_validate_feature,
                    validate_cache=validate_cache, X_test_feature=X_test_feature, test_cache=test_cache,
                    accuracy=accuracy_combination[i])
                time2 = time.time()
                self.train_time += (time2 - time1)
                retrained_pps_accuracy.append(flag)
                pp_filters_list.append(PPFilter(sub_plan=self.plan[i], estimator=prob_predictor.estimator,
                                                accuracy_info_dict=accuracy_info_dict,
                                                preprocessor=self.workflow.preprocessor,
                                                operator_name=str(self.plan) + "_" + str(i)))
                print("\t" + str(accuracy_info_dict))
        cost = self.compute_cost(pp_filter_list=pp_filters_list, ml_filter_list=self.ml_filters)
        acc_comb_pps = AccCombPPFiltersMap(accuracy_combination=accuracy_combination, cost=cost,
                                           pp_filters=pp_filters_list, retrain_flag=all(retrained_pps_accuracy))
        return acc_comb_pps

    def retrain_final_pps(self, result_accuracy_combination: List[float], first_pp: PPFilter):
        """
        final retrain PPs based on searched optimized accuracy combination
        :param result_accuracy_combination: searched optimized accuracy combination
        :param first_pp: the first pp which does not need to be retrained
        :return: pp_filters: List[PPFilter]: final validated searched optimized PPs
                 cost: float
                 final_accuracy: float
        """
        time1 = time.time()
        pp_filters = [first_pp]
        print("\t" + str(pp_filters[0].accuracy_info_dict))
        for i in range(1, len(result_accuracy_combination)):
            subplan_sample = self.plan_sample_labeler.subplan_samples[i].copy()
            for j in range(0, i):
                if self.workflow.workflow_type == UCF101_TYPE and self.workflow.opt_thread_num == 0:
                    pp_batch_output = pp_filters[j].process_batch(batch=subplan_sample, model=self.workflow.models[
                        self.workflow.preprocessor.operator_name])
                else:
                    pp_batch_output = pp_filters[j].multiple_threads_batch(batch=subplan_sample,
                                                                           num_process=self.workflow.opt_thread_num)
                subplan_sample = pp_batch_output.returned_batch
            X_train_feature, train_cache, X_validate_feature, validate_cache, X_test_feature, test_cache = \
                self.plan_pp_managers[i].data_set_convert(subplan_sample)
            time1 = time.time()
            proba_predicator = self.plan_pp_managers[i].retrain_pp(
                X_train_feature=X_train_feature, train_cache=train_cache, X_validate_feature=X_validate_feature,
                validate_cache=validate_cache, X_test_feature=X_test_feature, test_cache=test_cache)
            time2 = time.time()
            self.train_time += (time2 - time1)
            accuracy_info_dict = proba_predicator.accuracy_infos.get_accuracy_info(
                accuracy=result_accuracy_combination[i])
            print("\t" + str(accuracy_info_dict))
            pp_filters.append(PPFilter(sub_plan=self.plan[i], estimator=proba_predicator.estimator,
                                       accuracy_info_dict=accuracy_info_dict, preprocessor=self.workflow.preprocessor,
                                       operator_name=str(self.plan) + "_" + str(i)))
        cost = self.compute_cost(pp_filter_list=pp_filters, ml_filter_list=self.ml_filters)
        time2 = time.time()
        print("retrain time = " + str(time2 - time1))
        return pp_filters, cost
