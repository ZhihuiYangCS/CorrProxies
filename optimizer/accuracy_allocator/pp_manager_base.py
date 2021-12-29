import time
from typing import List
import numpy
import scipy
import rootpath

rootpath.append()
from ml_workflow.workflow_base import WorkflowBase
from utility.constant import TWITTER_PP_TYPE, UCF101_PP_TYPE
from optimizer.PPs.pp_trainer.train_dnn_classifier import TrainDNNClassifier
from optimizer.PPs.pp_trainer.train_svm_classifier import TrainSVMClassifier
from records.record import Record


class PPManagerBase:
    """
    a pp_manager to manage all history PPs for several ml_filters. These several ml_filters are indicated by sub_plan
    """

    def __init__(self, workflow: WorkflowBase, reuse_flag: bool = True):
        self.workflow = workflow
        self.reuse_flag = reuse_flag
        # List[ProbabilisticPredicator]
        self.pp_history = []
        self.f1_threshold = 0.005

    def data_set_convert(self, data_set: List[Record]):
        """
        convert a labeled_sample dataset to a format, which can be processed SVM classifier
        :param data_set: a list of record, labeled samples
        """
        time0 = time.time()
        X_train_feature = None
        train_ids = []
        y_train = []
        X_validate_feature = None
        validate_ids = []
        y_validate = []
        X_test_feature = None
        test_ids = []
        y_test = []
        for record in data_set:
            if record["split_flag"] == 0:
                if X_train_feature is None:
                    X_train_feature = self.workflow.preprocessor.get_result(record)
                else:
                    if isinstance(X_train_feature, numpy.ndarray):
                        X_train_feature = numpy.vstack((X_train_feature, self.workflow.preprocessor.get_result(record)))
                    else:
                        X_train_feature = scipy.sparse.vstack(
                            [X_train_feature, self.workflow.preprocessor.get_result(record)])
                train_ids.append(record.id)
                y_train.append(record.get_value("flag"))
            elif record["split_flag"] == 1:
                if X_validate_feature is None:
                    X_validate_feature = self.workflow.preprocessor.get_result(record)
                else:
                    if isinstance(X_validate_feature, numpy.ndarray):
                        X_validate_feature = numpy.vstack(
                            (X_validate_feature, self.workflow.preprocessor.get_result(record)))
                    else:
                        X_validate_feature = scipy.sparse.vstack(
                            [X_validate_feature, self.workflow.preprocessor.get_result(record)])
                validate_ids.append(record.id)
                y_validate.append(record.get_value("flag"))
            else:
                if X_test_feature is None:
                    X_test_feature = self.workflow.preprocessor.get_result(record)
                else:
                    if isinstance(X_test_feature, numpy.ndarray):
                        X_test_feature = numpy.vstack((X_test_feature, self.workflow.preprocessor.get_result(record)))
                    else:
                        X_test_feature = scipy.sparse.vstack(
                            [X_test_feature, self.workflow.preprocessor.get_result(record)])
                test_ids.append(record.id)
                y_test.append(record.get_value("flag"))
        train_cache = numpy.array([train_ids]).T
        train_cache = numpy.concatenate((train_cache, numpy.array([y_train]).T), axis=1)

        validate_cache = numpy.array([validate_ids]).T
        validate_cache = numpy.concatenate((validate_cache, numpy.array([y_validate]).T), axis=1)

        test_cache = numpy.array([test_ids]).T
        test_cache = numpy.concatenate((test_cache, numpy.array([y_test]).T), axis=1)
        time1 = time.time()
        print("data set convert from raw data to features; its cost is = " + str(time1 - time0))
        return X_train_feature, train_cache, X_validate_feature, validate_cache, X_test_feature, test_cache

    def retrain_pp(self, X_train_feature, train_cache, X_validate_feature, validate_cache, X_test_feature, test_cache):
        """
        retrain a pp
        :return: a retrained pp
        """
        if self.workflow.pp_type == TWITTER_PP_TYPE or (
                self.workflow.workflow_type == UCF101_PP_TYPE and self.workflow.opt_thread_num != 0):
            train_classifier = TrainSVMClassifier(X_train_feature=X_train_feature, train_cache=train_cache,
                                                  X_validate_feature=X_validate_feature,
                                                  validate_cache=validate_cache,
                                                  X_test_feature=X_test_feature, test_cache=test_cache)
        else:
            train_classifier = TrainDNNClassifier(X_train_feature=X_train_feature, train_cache=train_cache,
                                                  X_validate_feature=X_validate_feature,
                                                  validate_cache=validate_cache,
                                                  X_test_feature=X_test_feature, test_cache=test_cache)
        proba_predicator = train_classifier.train_classifier()
        self.pp_history.insert(0, proba_predicator)
        return proba_predicator

    def get_pp_and_accuracy_info(self, X_train_feature, train_cache, X_validate_feature, validate_cache, X_test_feature,
                                 test_cache, accuracy):
        """
        this method returns a pp for a new passed dataset and accuracy_info_dict for a specified accuracy
        :return: a pp (proba_predicator), a accuracy_reduction mapping (accuracy_info_dict)
                and a flag indicating retrain or not
        """
        if self.reuse_flag and self.pp_history:
            for proba_predicator in self.pp_history:
                f1_new_data = proba_predicator.compute_f1_score(X_train_feature=X_train_feature,
                                                                train_cache=train_cache)
                print("\ttwo pps f1 distance = " + str(abs(f1_new_data - proba_predicator.f1_train)))
                if abs(f1_new_data - proba_predicator.f1_train) < self.f1_threshold:
                    print("\treuse")
                    accuracy_info_dict = proba_predicator.compute_update_accuracy_info(accuracy=accuracy,
                                                                                       X_validate_feature=X_validate_feature,
                                                                                       validate_cache=validate_cache,
                                                                                       X_test_feature=X_test_feature,
                                                                                       test_cache=test_cache)
                    return proba_predicator, accuracy_info_dict, False
        proba_predicator = self.retrain_pp(X_train_feature=X_train_feature, train_cache=train_cache,
                                           X_validate_feature=X_validate_feature,
                                           validate_cache=validate_cache, X_test_feature=X_test_feature,
                                           test_cache=test_cache)
        accuracy_info_dict = proba_predicator.accuracy_infos.get_accuracy_info(accuracy=accuracy)
        return proba_predicator, accuracy_info_dict, True
