import time
from abc import ABC, abstractmethod
from typing import List
import rootpath

rootpath.append()
from utility.constant import TWITTER_TYPE, TWITTER_PP_TYPE, UCF101_PP_TYPE, UCF101_TYPE
from optimizer.accuracy_allocator.acc_allocator.accuracy_allocator_utility import AccCombPPFiltersMap
from ml_workflow.workflow_base import WorkflowBase
from operators.ml_filter import MLFilter
from optimizer.PPs.accuracy_infos import AccuracyInfoDict
from optimizer.PPs.pp_filter import PPFilter
from optimizer.PPs.pp_trainer.train_dnn_classifier import TrainDNNClassifier
from optimizer.PPs.pp_trainer.train_svm_classifier import TrainSVMClassifier
from optimizer.PPs.probabilistic_predicator import ProbabilisticPredicator
from utility.utility import sample_train_stop_condition, copy_samples, train_validate_test_split, data_set_convert, \
    batch_records_copy, get_selectivity


class MicrosoftPPBase(ABC):
    """
    A base class for Microsoft PP
    """

    def __init__(self, workflow: WorkflowBase):
        # self.workflow = workflow.get_new_workflow()
        self.workflow = workflow
        self.eta = 0.99
        self.ml_filters = self.workflow.get_ml_filters()
        self.prob_predictors = None
        self.labeled_samples = []
        self.ml_filter_selectivity = []
        # self.ml_operator_costs = None
        self.label_time = 0
        self.train_time = 0
        self.aa_time = 0
        self.reorder_time = 0

    def generate_labeled_samples(self):
        """
        generate labeled samples for each filter in the workflow
        """
        time1 = time.time()
        scan, sink = self.workflow.get_scan_sink()
        process_num = [0] * len(self.ml_filters)
        pass_num = [0] * len(self.ml_filters)
        labeled_sample_flags = [False] * len(self.ml_filters)
        scan.operator_reset()
        while not all(labeled_sample_flags):
            if scan.output_num % self.workflow.batch_size == 0:
                print(scan.output_num)
            batch_records = scan.get_block_records(self.workflow.batch_size)
            if self.workflow.workflow_type == UCF101_TYPE and self.workflow.opt_thread_num == 0:
                preprocessor_batch_output = self.workflow.preprocessor.process_batch(batch=batch_records,
                                                                                     model=self.workflow.models[
                                                                                         self.workflow.preprocessor.operator_name])
            else:
                preprocessor_batch_output = self.workflow.preprocessor.multiple_processes_batch(batch=batch_records,
                                                                                                num_process=self.workflow.opt_thread_num)
            batch_records = preprocessor_batch_output.returned_batch
            for i in range(len(self.ml_filters)):
                if not labeled_sample_flags[i]:
                    batch_records_labeled_samples = batch_records_copy(batch_records=batch_records)
                    if self.workflow.workflow_type == UCF101_TYPE and self.workflow.opt_thread_num == 0:
                        batch_output = self.ml_filters[i].process_batch(batch=batch_records, model=self.workflow.models[
                            self.ml_filters[i].ml_predicator.ml_operator.operator_name])
                    elif self.workflow.workflow_type == TWITTER_TYPE:
                        batch_output = self.ml_filters[i].multiple_threads_batch(batch=batch_records,
                                                                                 num_process=self.workflow.opt_thread_num,
                                                                                 model=self.workflow.models[
                                                                                     self.ml_filters[
                                                                                         i].ml_predicator.ml_operator.operator_name])
                    else:
                        batch_output = self.ml_filters[i].multiple_processes_batch(batch=batch_records,
                                                                                   num_process=self.workflow.opt_thread_num)
                    result_batch = batch_output.returned_batch
                    for index, one_labeled_sample in enumerate(batch_records_labeled_samples):
                        if self.ml_filters[i].ml_predicator.get_result(batch_records[index]):
                            pass_num[i] += 1
                            one_labeled_sample["flag"] = 1
                        else:
                            one_labeled_sample["flag"] = 0
                        process_num[i] += 1
                        if process_num[i] == 1:
                            self.labeled_samples.append([one_labeled_sample])
                        else:
                            self.labeled_samples[i].append(one_labeled_sample)
                        labeled_sample_flags[i] = sample_train_stop_condition(workflow_type=self.workflow.workflow_type,
                                                                              process_num=process_num[i],
                                                                              pos_num=pass_num[i])
                        if labeled_sample_flags[i]:
                            break
        time2 = time.time()
        self.label_time += (time2 - time1)
        print("scan.output_num = " + str(scan.output_num))
        for i in range(len(self.ml_filters)):
            print(str(i) + "\tlabeled_sample num = " + str(len(self.labeled_samples[i])) + "\tprocess_num = " + str(
                process_num[i]) + "\tpass_num = " + str(pass_num[i]) + "\tlabeled_sample_stop_flag = " + str(
                labeled_sample_flags[i]))
            self.labeled_samples[i] = copy_samples(workflow_type=self.workflow.workflow_type,
                                                   sample=self.labeled_samples[i])
            self.ml_filter_selectivity.append(get_selectivity(samples=self.labeled_samples[i]))
            train_validate_test_split(self.labeled_samples[i])
        print("ml_filters selectivity = " + str(self.ml_filter_selectivity))
        self.workflow.get_ml_operator_costs()

    def train_pps(self) -> List[ProbabilisticPredicator]:
        """
        train pps and get probabilistic_predicates for each filter using each filter's labeled sample
        """
        prob_predictors = []
        for i in range(len(self.labeled_samples)):
            X_train_feature, train_cache, X_validate_feature, validate_cache, X_test_feature, test_cache = \
                data_set_convert(self.labeled_samples[i], preprocessor=self.workflow.preprocessor)
            if self.workflow.pp_type == TWITTER_PP_TYPE or (
                    self.workflow.pp_type == UCF101_PP_TYPE and self.workflow.opt_thread_num != 0):
                train_classifier = TrainSVMClassifier(X_train_feature=X_train_feature, train_cache=train_cache,
                                                      X_validate_feature=X_validate_feature,
                                                      validate_cache=validate_cache,
                                                      X_test_feature=X_test_feature, test_cache=test_cache)
            else:
                train_classifier = TrainDNNClassifier(X_train_feature=X_train_feature, train_cache=train_cache,
                                                      X_validate_feature=X_validate_feature,
                                                      validate_cache=validate_cache,
                                                      X_test_feature=X_test_feature, test_cache=test_cache)
            prob_predictor = train_classifier.train_classifier()
            # for correlation experiments
            print("the " + str(i) + "th cascaded filter, its accuracy reduction curve:")
            print(prob_predictor.accuracy_infos)
            # for correlation experiments
            prob_predictors.append(prob_predictor)
        # for correlation experiments
        print("with previous accuracy fixed")
        pp_filters = [PPFilter(sub_plan=[0], estimator=prob_predictors[0].estimator,
                               accuracy_info_dict=prob_predictors[0].accuracy_infos.get_accuracy_info(accuracy=0.9),
                               preprocessor=self.workflow.preprocessor)]

        for i in range(1, len(self.labeled_samples)):
            labeled_samples = self.labeled_samples[i].copy()
            for pp_filter in pp_filters:
                if self.workflow.workflow_type == UCF101_TYPE and self.workflow.opt_thread_num == 0:
                    pp_batch_output = pp_filter.process_batch(batch=labeled_samples, model=self.workflow.models[
                        self.workflow.preprocessor.operator_name])
                else:
                    pp_batch_output = pp_filter.multiple_threads_batch(batch=labeled_samples,
                                                                       num_process=self.workflow.opt_thread_num)
                labeled_samples = pp_batch_output.returned_batch
                labeled_samples = copy_samples(workflow_type=self.workflow.workflow_type, sample=labeled_samples)
            X_train_feature, train_cache, X_validate_feature, validate_cache, X_test_feature, test_cache = \
                data_set_convert(labeled_samples, preprocessor=self.workflow.preprocessor)
            if self.workflow.pp_type == TWITTER_PP_TYPE or (
                    self.workflow.pp_type == UCF101_PP_TYPE and self.workflow.opt_thread_num != 0):
                train_classifier = TrainSVMClassifier(X_train_feature=X_train_feature, train_cache=train_cache,
                                                      X_validate_feature=X_validate_feature,
                                                      validate_cache=validate_cache,
                                                      X_test_feature=X_test_feature, test_cache=test_cache)
            else:
                train_classifier = TrainDNNClassifier(X_train_feature=X_train_feature, train_cache=train_cache,
                                                      X_validate_feature=X_validate_feature,
                                                      validate_cache=validate_cache,
                                                      X_test_feature=X_test_feature, test_cache=test_cache)
            prob_predictor = train_classifier.train_classifier()
            print("the " + str(i) + "th cascaded filter with previous accuracy fixed, its accuracy reduction curve:")
            print(prob_predictor.accuracy_infos)
            pp_filters.append(PPFilter(sub_plan=[i], estimator=prob_predictor.estimator,
                                       accuracy_info_dict=prob_predictor.accuracy_infos.get_accuracy_info(accuracy=0.9),
                                       preprocessor=self.workflow.preprocessor))
        # for correlation experiments
        return prob_predictors

    def compute_cost(self, accuracy_infos: List[AccuracyInfoDict], ml_filters: List[MLFilter]) -> float:
        """
        compute cost for each accuracy combination using their accuracy_infos
        :return: cost: float
        """
        r, c = 0, 0
        for i in range(len(accuracy_infos)):
            r, c = self.compute_cost_base(r, c, accuracy_infos[i].reduction, accuracy_infos[i].infer_cost)
        selectivity = 1
        u = 0
        operator_name = None
        for i in range(len(ml_filters)):
            if operator_name is None:
                operator_name = ml_filters[i].ml_predicator.ml_operator.operator_name
                u += selectivity * ml_filters[i].operator_cost
                selectivity *= self.ml_filter_selectivity[i]
            else:
                if operator_name == ml_filters[i].ml_predicator.ml_operator.operator_name:
                    selectivity *= self.ml_filter_selectivity[i]
                else:
                    operator_name = ml_filters[i].ml_predicator.ml_operator.operator_name
                    u += selectivity * ml_filters[i].operator_cost
                    selectivity *= self.ml_filter_selectivity[i]
        cost = c + (1 - r) * u
        return cost

    def compute_cost_base(self, a_reduction: float, a_infer_cost: float, b_reduction: float, b_infer_cost: float):
        """
        a base cost computation method
        """
        return a_reduction + b_reduction - a_reduction * b_reduction, min(
            a_infer_cost + (1 - a_reduction) * b_infer_cost, b_infer_cost + (1 - b_reduction) * a_infer_cost)

    def one_accuracy_combination_cost(self, accuracy_combination: List[float]) -> AccCombPPFiltersMap:
        """
        compute the cost for one accuracy_combination
        """
        accuracy_infos, pp_filters = [], []
        for i in range(len(accuracy_combination)):
            accuracy_info_dict = self.prob_predictors[i].accuracy_infos.get_accuracy_info(
                accuracy=accuracy_combination[i])
            accuracy_infos.append(accuracy_info_dict)
        cost = self.compute_cost(accuracy_infos=accuracy_infos, ml_filters=self.ml_filters)
        for i in range(len(self.ml_filters)):
            pp_filter = PPFilter(sub_plan=[i], estimator=self.prob_predictors[i].estimator,
                                 accuracy_info_dict=accuracy_infos[i],
                                 preprocessor=self.workflow.preprocessor)
            pp_filters.append(pp_filter)
        acc_comb_pps = AccCombPPFiltersMap(accuracy_combination=accuracy_combination, cost=cost,
                                           pp_filters=pp_filters, retrain_flag=True)
        print("cost = " + str(cost) + "\taccuracy_combinations = " + str(accuracy_combination))
        return acc_comb_pps

    @abstractmethod
    def search(self) -> List[PPFilter]:
        """
        :return:  -> List[PPFilter]
        """
        pass

    def multiple_pps_accuracy_allocation(self):
        """
        a outer interface to perform accuracy allocation using YaoLu's method
        :return: List[MLFilter]
        """
        time0 = time.time()
        self.generate_labeled_samples()
        time1 = time.time()
        print("generate_labeled_samples cost = " + str(time1 - time0))
        self.prob_predictors = self.train_pps()
        time2 = time.time()
        self.train_time += (time2 - time1)
        print("prob_predictors train cost = " + str(time2 - time1))
        pp_filters = self.search()
        time3 = time.time()
        self.aa_time += (time3 - time2)
        # self.close()
        print("search cost = " + str(time3 - time2))
        time_log = str(self.label_time) + "\t" + str(self.train_time) + "\t" + str(self.aa_time) + "\t" + str(
            self.reorder_time)
        return pp_filters, time_log
