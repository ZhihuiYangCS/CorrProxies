import time
from abc import ABC
from typing import List
import rootpath

rootpath.append()
from utility.constant import UCF101_TYPE
from optimizer.PPs.accuracy_infos import AccuracyInfoDict
from optimizer.PPs.pp_filter import PPFilter
from optimizer.accuracy_allocator.acc_allocator.accuracy_allocator_base import AccuracyAllocatorBase
from optimizer.accuracy_allocator.acc_allocator.accuracy_allocator_utility import AccCombPPFiltersMap
from records.record import Record
from utility.utility import get_selectivity


class AccuracyAllocatorCostFuncBase(AccuracyAllocatorBase, ABC):
    """
    exhausted search
    this method only works when plans are singlePP plan and multiplePPs plan.
    For example, the workflow is A(OP_1) -> B(OP_1) -> C(OP_2)
        singlePP plan is [[A,B,C]] and multiplePPs plan is [[A], [B], [C]]
        this method doesn't work for plan [[A],[B,C]]
    allocation_mode: 0 ---- resample retrain
                     1 ---- reuse sample, retrain
                     2 ---- reuse sample, reuse PPs
    """

    def __init__(self, workflow, plan: List[List[int]], workflow_samples: List[Record] = None):
        super().__init__(workflow=workflow, plan=plan, workflow_samples=workflow_samples)
        self.original_workflow_cost = None

    def compute_original_workflow_cost(self) -> float:
        """
        compute the original workflow's cost
        """
        original_workflow_cost = 0
        print("compute original workflow cost")
        ml_filters = self.workflow.get_ml_filters()
        original_selectivities = []
        for ml_filter in ml_filters:
            original_selectivities.append(ml_filter.get_operator_pass_rate())
        print("original_selectivities = " + str(original_selectivities))
        operator_name = None
        selectivity = 1
        for i in range(len(ml_filters)):
            if operator_name is None or ml_filters[i].ml_predicator.ml_operator.operator_name != operator_name:
                original_workflow_cost += ml_filters[i].operator_cost * selectivity
                selectivity *= original_selectivities[i]
                operator_name = ml_filters[i].ml_predicator.ml_operator.operator_name
            else:
                selectivity *= original_selectivities[i]
        return original_workflow_cost

    def compute_cost(self, accuracy_infos: List[AccuracyInfoDict], updated_selectivities: List[float]) -> float:
        """
        this method doesn't work when plans are not singlePP and multiplePPs
        :return: cost
        """
        ml_filters = self.workflow.get_ml_filters()
        if len(accuracy_infos) == 1:
            save_time = accuracy_infos[0].reduction * ml_filters[0].ml_predicator.ml_operator.operator_cost - \
                        accuracy_infos[0].infer_cost
        else:
            save_time = 0
            operator_name = None
            pass_rate = 1
            last_unit_pass_rate = 1
            for i in range(len(accuracy_infos)):
                if operator_name is None or (operator_name is not None and operator_name != ml_filters[
                    i].ml_predicator.ml_operator.operator_name):
                    pass_rate = last_unit_pass_rate
                operator_name = ml_filters[i].ml_predicator.ml_operator.operator_name
                save_time += pass_rate * (
                        accuracy_infos[i].reduction * ml_filters[i].ml_predicator.ml_operator.operator_cost -
                        accuracy_infos[i].infer_cost)
                last_unit_pass_rate *= pass_rate * updated_selectivities[i] * accuracy_infos[i].target_accuracy
                pass_rate *= (1 - accuracy_infos[i].reduction)
        print("orginal_workflow_cost = " + str(self.original_workflow_cost) + "\tsave_time = " + str(
            save_time) + "\tcost = " + str(self.original_workflow_cost - save_time) + "\tupdated_selectivity = " + str(
            updated_selectivities))
        return self.original_workflow_cost - save_time

    def one_accuracy_combination_cost(self, accuracy_combination: List[float]) -> AccCombPPFiltersMap:
        """
        compute the cost for one accuracy combination
        :param accuracy_combination: an accuracy combination to compute the cost
        :return: an accuracy_combination_pp_filters_map
        """
        self._get_samples()
        for ml_filter in self.ml_filters:
            ml_filter.get_operator_cost()
            if ml_filter is None:
                ml_filter.print_operator_detail()
                exit(1)
        self.original_workflow_cost = self.compute_original_workflow_cost()
        pp_filters_list = []
        retrained_pps_accuracy = []
        accuracy_info_list = []
        updated_selectivities = []
        for i in range(len(accuracy_combination)):
            sub_plan_sample = self.plan_sample_labeler.subplan_samples[i].copy()
            if i == 0 and self.plan_pp_managers[i].pp_history:
                accuracy_info_dict = self.plan_pp_managers[i].pp_history[0].accuracy_infos.get_accuracy_info(
                    accuracy=accuracy_combination[i])
                accuracy_info_list.append(accuracy_info_dict)
                updated_selectivities.append(get_selectivity(samples=sub_plan_sample))
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
                updated_selectivities.append(get_selectivity(samples=sub_plan_sample))
                print("\tsub_plan_sample num = " + str(len(sub_plan_sample)))
                X_train_feature, train_cache, X_validate_feature, validate_cache, X_test_feature, test_cache = \
                    self.plan_pp_managers[i].data_set_convert(sub_plan_sample)
                time1 = time.time()
                proba_predicator, accuracy_info_dict, flag = self.plan_pp_managers[i].get_pp_and_accuracy_info(
                    X_train_feature=X_train_feature, train_cache=train_cache, X_validate_feature=X_validate_feature,
                    validate_cache=validate_cache, X_test_feature=X_test_feature, test_cache=test_cache,
                    accuracy=accuracy_combination[i])
                time2 = time.time()
                self.train_time += (time2 - time1)
                retrained_pps_accuracy.append(flag)
                accuracy_info_list.append(accuracy_info_dict)
                pp_filters_list.append(PPFilter(sub_plan=self.plan[i], estimator=proba_predicator.estimator,
                                                accuracy_info_dict=accuracy_info_dict,
                                                preprocessor=self.workflow.preprocessor,
                                                operator_name=str(self.plan) + "_" + str(i)))
                print("\t" + str(accuracy_info_dict))
        cost = self.compute_cost(accuracy_infos=accuracy_info_list, updated_selectivities=updated_selectivities)
        acc_comb_pps = AccCombPPFiltersMap(accuracy_combination=accuracy_combination, cost=cost,
                                           pp_filters=pp_filters_list, retrain_flag=all(retrained_pps_accuracy))
        return acc_comb_pps

    def retrain_final_pps(self, result_accuracy_combination: List[float], first_pp: PPFilter):
        """
        final retrain PPs based on searched optimized accuracy combination
        :param original_workflow_cost:
        :param ml_operator_costs:
        :param result_accuracy_combination: searched optimized accuracy combination
        :param first_pp: the first pp which does not need to be retrained
        :return: pp_filters: List[PPFilter]: final validated searched optimized PPs
                 cost: float
                 final_accuracy: float
        """
        pp_filters = [first_pp]
        updated_selectivities = [get_selectivity(self.plan_sample_labeler.subplan_samples[0])]
        accuracy_info_list = [pp_filters[0].accuracy_info_dict]
        print("\t" + str(pp_filters[0].accuracy_info_dict))
        for i in range(1, len(result_accuracy_combination)):
            sub_plan_sample = self.plan_sample_labeler.subplan_samples[i].copy()
            for j in range(0, i):
                if self.workflow.workflow_type == UCF101_TYPE and self.workflow.opt_thread_num == 0:
                    pp_batch_output = pp_filters[j].process_batch(batch=sub_plan_sample, model=self.workflow.models[
                        self.workflow.preprocessor.operator_name])
                else:
                    pp_batch_output = pp_filters[j].multiple_threads_batch(batch=sub_plan_sample,
                                                                           num_process=self.workflow.opt_thread_num)
                sub_plan_sample = pp_batch_output.returned_batch
            updated_selectivities.append(get_selectivity(samples=sub_plan_sample))
            X_train_feature, train_cache, X_validate_feature, validate_cache, X_test_feature, test_cache = \
                self.plan_pp_managers[i].data_set_convert(sub_plan_sample)
            time1 = time.time()
            prob_predictor = self.plan_pp_managers[i].retrain_pp(
                X_train_feature=X_train_feature, train_cache=train_cache, X_validate_feature=X_validate_feature,
                validate_cache=validate_cache, X_test_feature=X_test_feature, test_cache=test_cache)
            time2 = time.time()
            self.train_time += (time2 - time1)
            accuracy_info_dict = prob_predictor.accuracy_infos.get_accuracy_info(
                accuracy=result_accuracy_combination[i])
            print("\t" + str(accuracy_info_dict))
            accuracy_info_list.append(accuracy_info_dict)
            pp_filters.append(PPFilter(sub_plan=self.plan[i], estimator=prob_predictor.estimator,
                                       accuracy_info_dict=accuracy_info_dict, preprocessor=self.workflow.preprocessor,
                                       operator_name=str(self.plan) + "_" + str(i)))
        cost = self.compute_cost(accuracy_infos=accuracy_info_list,
                                 updated_selectivities=updated_selectivities)
        return pp_filters, cost
