import time
from abc import ABC, abstractmethod
from sklearn.metrics import f1_score
import numpy
import rootpath

rootpath.append()
from optimizer.PPs.accuracy_infos import AccuracyInfoDict, AccuracyInfos
from optimizer.PPs.probabilistic_predicator import ProbabilisticPredicator


class TrainClassifier(ABC):
    def __init__(self, X_train_feature, train_cache, X_validate_feature, validate_cache, X_test_feature, test_cache):
        self.X_train_feature = X_train_feature
        self.train_cache = train_cache
        self.X_validate_feature = X_validate_feature
        self.validate_cache = validate_cache
        self.X_test_feature = X_test_feature
        self.test_cache = test_cache

        self.pos_num = numpy.sum(self.train_cache[:, 1] == 1) + numpy.sum(self.validate_cache[:, 1] == 1) + numpy.sum(
            self.test_cache[:, 1] == 1)
        self.neg_num = numpy.sum(self.train_cache[:, 1] == 0) + numpy.sum(self.validate_cache[:, 1] == 0) + numpy.sum(
            self.test_cache[:, 1] == 0)
        print("\tpos_num = " + str(self.pos_num) + "\tneg_num = " + str(self.neg_num))

    def my_own_score(self, estimator, X, y_true):
        """
        my own scoring function for grid search.
        It computes the estimator's accuracy_reduction mapping relationship.
        If the abs of target_accuracy-final_accuracy is less than 0.01:
            its score added the reducion
        else:
            its score add 0.
        :param estimator:
        :param X:
        :param y_true:
        :return:
        """
        numpy.random.seed(1)
        index = numpy.arange(numpy.shape(X)[0])
        numpy.random.shuffle(index)

        X_validate, y_true_validate = X[index[:int(len(index) * 0.5)], :], numpy.take(y_true,
                                                                                      index[:int(len(index) * 0.5)])
        X_test, y_true_test = X[index[int(len(index) * 0.5):], :], numpy.take(y_true, index[int(len(index) * 0.5):])

        pos_class_index = int(numpy.where(estimator.classes_ == 1)[0][0])

        y_pred_proba = estimator.predict_proba(X_validate)[:, pos_class_index]
        pos_proba = y_pred_proba[y_true_validate == 1]
        pos_proba_sorted = -numpy.sort(-pos_proba)

        y_pred_test = estimator.predict_proba(X_test)[:, pos_class_index]
        denominator_accuracy = numpy.sum(y_true_test == 1)

        target_accuracy = [1.0, 0.98, 0.96, 0.94, 0.92, 0.9]
        score = []
        for accuracy in target_accuracy:
            threshold = pos_proba_sorted[int(pos_proba_sorted.shape[0] * accuracy) - 1]
            y_pos_test = numpy.copy(y_pred_test)
            y_pos_test[y_pos_test >= threshold] = 1
            y_pos_test[y_pos_test < threshold] = 0
            denominator_reduction = y_pos_test.shape[0]
            nominator_reduction = list(y_pos_test).count(0)
            y_pos_test = y_pos_test[y_true_test == 1]
            nominator_accuracy = list(y_pos_test).count(1)
            final_accuracy = nominator_accuracy / denominator_accuracy
            reduction = nominator_reduction / denominator_reduction
            if abs(final_accuracy - accuracy) <= 0.015:
                score.append(reduction)
            else:
                score.append(0)
        return sum(score)

    @abstractmethod
    def train_classifier_grid_search_cross_validate(self):
        pass

    def compute_accuracy_reduction_relationship(self, estimator, train_cost):
        """
        compute the accuracy reduction mapping relationship,
        and update validate_cache and test_cache by adding y_pred_proba
        :param estimator: a classifier
        :param train_cost: float
        :return: AccuracyInfos:List[AccuracyInfoDict]
        """
        pos_class_index = int(numpy.where(estimator.classes_ == 1)[0][0])
        accuracy_info_list = []
        time_1 = time.time()
        y_validate_proba = estimator.predict_proba(self.X_validate_feature)[:, pos_class_index]
        time_2 = time.time()
        infer_cost = (time_2 - time_1) / self.validate_cache.shape[0]
        self.validate_cache = numpy.concatenate(
            (self.validate_cache, y_validate_proba.reshape((y_validate_proba.shape[0], 1))), axis=1)
        pos_proba_validate = y_validate_proba[self.validate_cache[:, 1] == 1]
        pos_proba_sorted = -numpy.sort(-pos_proba_validate)

        index = 0
        y_pred_test = estimator.predict_proba(self.X_test_feature)[:, pos_class_index]
        self.test_cache = numpy.concatenate((self.test_cache, y_pred_test.reshape((y_pred_test.shape[0], 1))), axis=1)
        denominator_accuracy = numpy.sum(self.test_cache[:, 1] == 1)
        for i in range(pos_proba_sorted.shape[0] - 1, -1, -1):  # loop decending order [0,pos_proba_sorted.shape[0]-1]
            threshold = pos_proba_sorted[i]
            target_accuracy = (i + 1) / (pos_proba_sorted.shape[0])
            if (i + 1) / (pos_proba_sorted.shape[0]) < 0.9:
                break
            y_pos_test = numpy.copy(y_pred_test)
            y_pos_test[y_pos_test >= threshold] = 1
            y_pos_test[y_pos_test < threshold] = 0
            denominator_reduction = y_pos_test.shape[0]
            nominator_reduction = list(y_pos_test).count(0)
            y_pos_test = y_pos_test[self.test_cache[:, 1] == 1]
            nominator_accuracy = list(y_pos_test).count(1)
            final_accuracy = nominator_accuracy / denominator_accuracy
            reduction = nominator_reduction / denominator_reduction
            accuracy_info_dict = AccuracyInfoDict(index=index, train_cost=train_cost, infer_cost=infer_cost,
                                                  target_accuracy=target_accuracy, final_accuracy=final_accuracy,
                                                  threshold=threshold, reduction=reduction)
            index += 1
            accuracy_info_list.append(accuracy_info_dict)
        accuracy_info = AccuracyInfos(accuracy_info_list)
        return accuracy_info

    def compute_estimator_f1(self, estimator):
        """
        compute the estimator's f1_score on the training set, and update train_cache by add y_pred
        :param estimator: a classifier
        :return: f1_score on training set
        """
        y_pred_train = estimator.predict(self.X_train_feature)
        self.train_cache = numpy.concatenate((self.train_cache, y_pred_train.reshape((y_pred_train.shape[0], 1))),
                                             axis=1)
        f1_train = f1_score(self.train_cache[:, 1], y_pred_train, average='weighted')
        return f1_train

    def train_classifier(self):
        """
        the whole procedure to train a PP
        :return: a PP (ProbabilisticPredicator)
        """
        estimator, train_cost = self.train_classifier_grid_search_cross_validate()
        accuracy_infos = self.compute_accuracy_reduction_relationship(estimator=estimator, train_cost=train_cost)
        # print(accuracy_infos.accuracy_info_list[-1])
        f1_train = self.compute_estimator_f1(estimator)
        print("\tf1_train = " + str(f1_train))
        proba_predicator = ProbabilisticPredicator(estimator=estimator, accuracy_infos=accuracy_infos,
                                                   train_cache=self.train_cache, validate_cache=self.validate_cache,
                                                   test_cache=self.test_cache, f1_train=f1_train)
        return proba_predicator
