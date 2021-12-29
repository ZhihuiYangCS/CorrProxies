import time
import warnings

warnings.filterwarnings('ignore')
import numpy
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.svm import SVC
import rootpath

rootpath.append()
from optimizer.PPs.pp_trainer.train_classifier import TrainClassifier


class TrainSVMClassifier(TrainClassifier):
    """
    This class trains a PP for input data set
    """

    def __init__(self, X_train_feature, train_cache, X_validate_feature, validate_cache, X_test_feature, test_cache):
        """
        pass in the train, validate, test data sets,
        compute pos_num and neg_num
        initialize search parameters: param_grid
        :param X_train_feature: feature for training data set
        :param train_cache: id, y_true for training data set
        :param X_validate_feature: feature for validate data set
        :param validate_cache: id, y_true for validate data set
        :param X_test_feature: feature for test data set
        :param test_cache: id, y_true for test data set
        """
        super().__init__(X_train_feature=X_train_feature, train_cache=train_cache,
                         X_validate_feature=X_validate_feature, validate_cache=validate_cache,
                         X_test_feature=X_test_feature, test_cache=test_cache)
        self.param_grid = {'C': numpy.array([0.01, 0.1, 1]), 'class_weight': numpy.array(
            [{0: 1, 1: self.neg_num / self.pos_num * 0.1}, {0: 1, 1: self.neg_num / self.pos_num * 1},
             {0: 1, 1: self.neg_num / self.pos_num * 5}, {0: 1, 1: self.neg_num / self.pos_num * 10}])}

    def train_svm(self):
        """
        train a svm classifier:
            we first select parameters using GridSearchCV with 'f1-weighted'
            then we train a classifier using CrossValidate with the selected parameters
        :return: estimator, train_cost
        """
        time_1 = time.time()
        grid_search = GridSearchCV(SVC(kernel='linear', tol=1e-5, probability=True, random_state=1), self.param_grid,
                                   cv=3, scoring='f1_weighted', n_jobs=-1)
        grid_search.fit(self.X_train_feature, self.train_cache[:, 1])
        best_params = grid_search.best_params_
        cv = cross_validate(SVC(C=best_params['C'], kernel='linear', tol=1e-5, probability=True, random_state=1,
                                class_weight=best_params['class_weight']), X=self.X_train_feature,
                            y=self.train_cache[:, 1], cv=3, scoring='f1_weighted', return_train_score=True,
                            return_estimator=True, n_jobs=-1)
        estimator = cv["estimator"][numpy.argmax(cv["test_score"])]
        time_2 = time.time()
        train_cost = (time_2 - time_1)
        return estimator, train_cost

    def train_classifier_grid_search_cross_validate(self):
        """
        train a svm classifier:
            we first select parameters using GridSearchCV with 'my_own_score'
            then we train a classifier using CrossValidate with the selected parameters
        :return: estimator, train_cost
        """
        time_1 = time.time()
        scoring = {'f1_weighted': 'f1_weighted', 'reduction': self.my_own_score}
        grid_search = GridSearchCV(SVC(kernel='linear', tol=1e-5, probability=True, random_state=1), self.param_grid,
                                   cv=3, scoring=scoring, refit='f1_weighted', n_jobs=-1)
        grid_search.fit(self.X_train_feature, self.train_cache[:, 1])
        best_params = grid_search.best_params_
        cv = cross_validate(SVC(C=best_params['C'], kernel='linear', tol=1e-5, probability=True, random_state=1,
                                class_weight=best_params['class_weight']), X=self.X_train_feature,
                            y=self.train_cache[:, 1], cv=3, scoring=scoring, return_estimator=True, n_jobs=-1)
        estimator = cv["estimator"][numpy.argmax(cv["test_f1_weighted"])]
        time_2 = time.time()
        train_cost = (time_2 - time_1)
        return estimator, train_cost
