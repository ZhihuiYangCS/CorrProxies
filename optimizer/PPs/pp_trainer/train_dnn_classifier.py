import os
import time
import warnings

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import logging

logging.getLogger('tensorflow').disabled = True
import numpy
from sklearn.utils import shuffle
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_validate, GridSearchCV
import rootpath

rootpath.append()
from optimizer.PPs.pp_trainer.train_classifier import TrainClassifier


class TrainDNNClassifier(TrainClassifier):
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
        epochs = [10, 20]
        self.param_grid = dict(epochs=epochs)

    def my_own_score_f1(self, estimator, X, y_true):
        y_pred = self.label_convert_prob(estimator.predict(X))
        return f1_score(y_true=y_true, y_pred=y_pred, average='weighted')

    def get_model(self):
        """
        define a the model structure
        nd: input dimension for each example, the number of features for each input
        nl: the dimension of the output label, the number of output labels
        :return: a model
        """
        nd: int = self.X_train_feature.shape[1]
        nl: int = 2
        model = Sequential()
        model.add(Dense(128, activation='sigmoid', input_shape=(nd,)))
        model.add(Dense(nl, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
        return model

    def label_convert_prob(self, labels):
        result = []
        [rows, ] = labels.shape
        for i in range(rows):
            if labels[i] == 1:
                result.append([0, 1])
            else:
                result.append([1, 0])
        return numpy.array(result)

    def training_set_balancing(self):
        zero_index = numpy.argwhere(self.train_cache[:, 1] == 0).T[0]
        zero_num = zero_index.shape[0]
        zero_train_cache = self.train_cache[zero_index, :]
        zero_train_cache_copy = zero_train_cache
        zero_features = self.X_train_feature[zero_index, :]
        zero_features_copy = zero_features

        one_index = numpy.argwhere(self.train_cache[:, 1] == 1).T[0]
        one_num = one_index.shape[0]
        one_train_cache = self.train_cache[one_index, :]
        one_train_cache_copy = one_train_cache
        one_features = self.X_train_feature[one_index, :]
        one_features_copy = one_features
        zero_sub_index, one_sub_index = 0, 0
        while zero_num != one_num:
            if zero_num < one_num:
                zero_features_copy = numpy.append(zero_features_copy, numpy.array([zero_features[zero_sub_index, :]]),
                                                  axis=0)
                zero_train_cache_copy = numpy.append(zero_train_cache_copy,
                                                     numpy.array([zero_train_cache[zero_sub_index, :]]), axis=0)
                zero_num += 1
                zero_sub_index += 1
                if zero_sub_index == zero_features.shape[0]:
                    zero_sub_index = 0
            if zero_num > one_num:
                one_features_copy = numpy.append(one_features_copy, numpy.array([one_features[one_sub_index, :]]),
                                                 axis=0)
                one_train_cache_copy = numpy.append(one_train_cache_copy,
                                                    numpy.array([one_train_cache[one_sub_index, :]]), axis=0)
                one_num += 1
                one_sub_index += 1
                if one_sub_index == one_features.shape[0]:
                    one_sub_index = 0
        train_cache = numpy.append(zero_train_cache_copy, one_train_cache_copy, axis=0)
        train_features = numpy.append(zero_features_copy, one_features_copy, axis=0)
        # print("train_cache shape = " + str(train_cache.shape) + "train_features shape = " + str(train_features.shape))
        self.X_train_feature, self.train_cache = shuffle(train_features, train_cache, random_state=1)
        print("X_train_feature shape = " + str(self.X_train_feature.shape) + "\ttrain_cache shape = " + str(
            self.train_cache.shape))

    def train_classifier_grid_search_cross_validate(self):
        self.training_set_balancing()
        time_1 = time.time()
        scoring = {'f1_weighted': self.my_own_score_f1, 'reduction': self.my_own_score}

        model = KerasClassifier(build_fn=self.get_model, batch_size=32, shuffle=True, verbose=0)
        grid_search = GridSearchCV(model, param_grid=self.param_grid, cv=3, scoring=scoring, refit='f1_weighted')
        grid_search.fit(self.X_train_feature, self.label_convert_prob(self.train_cache[:, 1]))
        # print(grid_search.cv_results_)
        best_params = grid_search.best_params_
        print("best_params = " + str(best_params))
        best_model = KerasClassifier(build_fn=self.get_model, batch_size=32, epochs=best_params['epochs'], shuffle=True,
                                     verbose=0)
        cv = cross_validate(best_model, X=self.X_train_feature, y=self.label_convert_prob(self.train_cache[:, 1]), cv=3,
                            scoring=scoring, return_estimator=True)
        # print("Cross Validation Scores = ")
        # print(cv)
        estimator = cv["estimator"][numpy.argmax(cv["test_f1_weighted"])]
        time_2 = time.time()
        train_cost = (time_2 - time_1)
        return estimator, train_cost

    def pro_to_label(self, pred_prob):
        result = []
        [rows, cols] = pred_prob.shape
        for i in range(rows):
            prob_distribution = pred_prob[i]
            result.append(numpy.argmax(prob_distribution))
        return numpy.array(result)
