import numpy
import rootpath
from sklearn.metrics import f1_score

rootpath.append()
from optimizer.PPs.accuracy_infos import AccuracyInfos, AccuracyInfoDict


class ProbabilisticPredicator:
    """
    this class encapsulate a PP, which has
        estimator: a classifier,
        accuracy_infos: the accuracy, reduction mapping relationship
        train_cache: Its format is 'id, y_true, y_pred', storing all predicted labels for reusing
        validate_cache, test_cache: Their format are 'id, y_true, y_pred_proba',
                                    storing all predicted probability on positive label for reusing
        f1_train: the estimator's f1_score on training data.
    """

    def __init__(self, estimator, accuracy_infos: AccuracyInfos, train_cache, validate_cache, test_cache,
                 f1_train: float):
        self.estimator = estimator
        self.accuracy_infos = accuracy_infos
        self.train_cache = train_cache
        self.validate_cache = validate_cache
        self.test_cache = test_cache
        self.f1_train = f1_train

    def compute_f1_score(self, X_train_feature, train_cache):
        """
        compute the PP's estimator's f1_score on new training data set, and update the PP's train_cache for reusing
        this function is used to measure the reuse possibility
        :param X_train_feature: text feature of the new training data set
        :param train_cache: id and y_true cache of the new training data set
        :return: f1_train: the PP's estimator's f1_score on the new training set
        """
        new_train_cache = self._update_cache(X_feature=X_train_feature, cache=train_cache, self_cache=self.train_cache,
                                             predict_proba=False)
        f1_train = f1_score(new_train_cache[:, 1], new_train_cache[:, 2], average='weighted')
        return f1_train

    def _update_cache(self, X_feature, cache, self_cache, predict_proba=True):
        """
        an inner function to update cache
        :param X_feature: a new text feature
        :param cache: a new cache
        :param self_cache: self.train_cache, self.validate_cache or self.test_cache
        :param predict_proba: whether to predict probability or not
        :return: a new cache
        """
        pos_class_index = int(numpy.where(self.estimator.classes_ == 1)[0][0])
        diff = numpy.invert(numpy.in1d(cache[:, 0], self_cache[:, 0]))
        cache_diff = cache[diff, :]
        if cache_diff.shape[0] != 0:
            feature_diff = X_feature[diff, :]
            if predict_proba:
                pred_diff = self.estimator.predict_proba(feature_diff)[:, pos_class_index]
            else:
                pred_diff = self.estimator.predict(feature_diff)
            cache_diff = numpy.concatenate(
                (cache_diff, pred_diff.reshape((pred_diff.shape[0], 1))), axis=1)
            self_cache = numpy.concatenate((self_cache, cache_diff), axis=0)

        new_cache = numpy.copy(self_cache[numpy.in1d(self_cache[:, 0], cache[:, 0])])
        return new_cache

    def compute_update_accuracy_info(self, accuracy, X_validate_feature, validate_cache, X_test_feature, test_cache):
        """
        compute an updated accuracy_info_dict for a specified accuracy on a new data set
        this function is used when we reuse this PP
        :param accuracy: a specified accuracy
        :return: a accuracy_info_dict
        """
        new_validate_cache = self._update_cache(X_feature=X_validate_feature, cache=validate_cache,
                                                self_cache=self.validate_cache)
        pos_proba_validate = new_validate_cache[new_validate_cache[:, 1] == 1][:, 2]
        pos_proba_sorted = -numpy.sort(-pos_proba_validate)

        index = 0
        train_cost = self.accuracy_infos.accuracy_info_list[0].train_cost
        infer_cost = self.accuracy_infos.accuracy_info_list[0].infer_cost
        threshold = pos_proba_sorted[int(pos_proba_sorted.shape[0] * accuracy) - 1]

        new_test_cache = self._update_cache(X_feature=X_test_feature, cache=test_cache, self_cache=self.test_cache)
        denominator_accuracy = numpy.sum(new_test_cache[:, 1] == 1)
        y_pred_test = numpy.copy(new_test_cache[:, 2])

        y_pred_test[y_pred_test >= threshold] = 1
        y_pred_test[y_pred_test < threshold] = 0
        denominator_reduction = y_pred_test.shape[0]
        nominator_reduction = list(y_pred_test).count(0)
        y_pos_test = y_pred_test[new_test_cache[:, 1] == 1]
        nominator_accuracy = list(y_pos_test).count(1)
        final_accuracy = nominator_accuracy / denominator_accuracy
        reduction = nominator_reduction / denominator_reduction
        accuracy_info_dict = AccuracyInfoDict(index=index, train_cost=train_cost, infer_cost=infer_cost,
                                              target_accuracy=accuracy, final_accuracy=final_accuracy,
                                              threshold=threshold, reduction=reduction)
        return accuracy_info_dict
