import random
from typing import List
import numpy
import scipy
import rootpath


rootpath.append()
from operators.operator_base.operator_parallel import OperatorParallel
from optimizer.PPs.pp_trainer.train_dnn_classifier import TrainDNNClassifier
from optimizer.PPs.pp_trainer.train_svm_classifier import TrainSVMClassifier
from optimizer.PPs.preprocessor.preprocessor import Preprocessor
from records.record import Record
from utility.constant import TWITTER_TYPE, TWITTER_SAMPLE_SIZE, COCO_TYPE, COCO_SAMPLE_SIZE, UCF101_SAMPLE_SIZE, \
    TWITTER_SELECT_SIZE, COCO_SELECT_SIZE, UCF101_SELECT_SIZE, TWITTER_SAMPLE_COPY_SIZE, COCO_SAMPLE_COPY_SIZE, \
    UCF101_SAMPLE_COPY_SIZE, TWITTER_PP_TYPE


def batch_records_delete_ml_operator_results(batch_records: List[Record], ml_operators: List[OperatorParallel]):
    """
    delete ml operators results on records
    """
    for record in batch_records:
        for ml_operator in ml_operators:
            record.del_key_value(ml_operator.get_result_key())


def sample_train_stop_condition(workflow_type: int, process_num: int, pos_num: int) -> bool:
    """
    the condition to stop getting samples for training:
        when sample_size >= 9000 and pos_num >= 2000 and neg_num >= 2000, stop
    """
    if workflow_type == TWITTER_TYPE:
        # if process_num >= 10000 and pos_num >= 2000 and (process_num - pos_num) >= 2000:
        if process_num >= TWITTER_SAMPLE_SIZE and pos_num >= TWITTER_SAMPLE_SIZE / 10 and (
                process_num - pos_num) >= TWITTER_SAMPLE_SIZE / 10:
            # if process_num >= 10000 and pos_num >= 1000 and (process_num - pos_num) >= 1000:
            return True
        else:
            return False
    elif workflow_type == COCO_TYPE:
        if process_num >= COCO_SAMPLE_SIZE and pos_num >= COCO_SAMPLE_SIZE / 10 and (
                process_num - pos_num) >= COCO_SAMPLE_SIZE / 10:
            return True
        else:
            return False
    else:
        if process_num >= UCF101_SAMPLE_SIZE and pos_num >= UCF101_SAMPLE_SIZE / 10 and (
                process_num - pos_num) >= UCF101_SAMPLE_SIZE / 10:
            return True
        else:
            return False


def sample_train_stop_condition_1(workflow_type: int, labeled_samples: List[Record]):
    """
    the condition to stop getting samples for training:
        when sample_size >= 9000 and pos_num >= 2000 and neg_num >= 2000, stop
    """
    process_num, pos_num = len(labeled_samples), 0
    for record in labeled_samples:
        if record["flag"] == 1:
            pos_num += 1
    return sample_train_stop_condition(workflow_type=workflow_type, process_num=process_num, pos_num=pos_num)


def sample_selectivity_stop_condition(workflow_type: int, process_num: int, pos_num: int) -> bool:
    """
    the condition to stop getting samples for selectivity
    """
    if workflow_type == TWITTER_TYPE:
        if process_num >= TWITTER_SELECT_SIZE and pos_num >= TWITTER_SELECT_SIZE / 100 and (
                process_num - pos_num) >= TWITTER_SELECT_SIZE / 100:
            return True
        else:
            return False
    elif workflow_type == COCO_TYPE:
        if process_num >= COCO_SELECT_SIZE and pos_num >= COCO_SELECT_SIZE / 100 and (
                process_num - pos_num) >= COCO_SELECT_SIZE / 100:
            return True
        else:
            return False
    else:
        if process_num >= UCF101_SELECT_SIZE and pos_num >= UCF101_SELECT_SIZE / 100 and (
                process_num - pos_num) >= UCF101_SELECT_SIZE / 100:
            return True
        else:
            return False


def sample_selectivity_stop_condition_1(workflow_type: int, labeled_samples: List[Record]):
    """
    the condition to stop getting samples for selectivity
    """
    process_num, pos_num = len(labeled_samples), 0
    for record in labeled_samples:
        if record["flag"] == 1:
            pos_num += 1
    return sample_selectivity_stop_condition(workflow_type=workflow_type, process_num=process_num, pos_num=pos_num)


def data_set_convert(data_set: List[Record], preprocessor: Preprocessor):
    """
    convert a labeled_sample dataset to a format, which can be processed SVM classifier
    :param preprocessor: a preprocessor convert unstructured text, image, video into matrix
    :param data_set: a list of record, labeled samples
    """
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
                X_train_feature = preprocessor.get_result(record)
            else:
                if isinstance(X_train_feature, numpy.ndarray):
                    X_train_feature = numpy.vstack((X_train_feature, preprocessor.get_result(record)))
                else:
                    X_train_feature = scipy.sparse.vstack([X_train_feature, preprocessor.get_result(record)])
            train_ids.append(record.id)
            y_train.append(record.get_value("flag"))
        elif record["split_flag"] == 1:
            if X_validate_feature is None:
                X_validate_feature = preprocessor.get_result(record)
            else:
                if isinstance(X_validate_feature, numpy.ndarray):
                    X_validate_feature = numpy.vstack((X_validate_feature, preprocessor.get_result(record)))
                else:
                    X_validate_feature = scipy.sparse.vstack(
                        [X_validate_feature, preprocessor.get_result(record)])
            validate_ids.append(record.id)
            y_validate.append(record.get_value("flag"))
        else:
            if X_test_feature is None:
                X_test_feature = preprocessor.get_result(record)
            else:
                if isinstance(X_test_feature, numpy.ndarray):
                    X_test_feature = numpy.vstack((X_test_feature, preprocessor.get_result(record)))
                else:
                    X_test_feature = scipy.sparse.vstack([X_test_feature, preprocessor.get_result(record)])
            test_ids.append(record.id)
            y_test.append(record.get_value("flag"))
    train_cache = numpy.array([train_ids]).T
    train_cache = numpy.concatenate((train_cache, numpy.array([y_train]).T), axis=1)

    validate_cache = numpy.array([validate_ids]).T
    validate_cache = numpy.concatenate((validate_cache, numpy.array([y_validate]).T), axis=1)

    test_cache = numpy.array([test_ids]).T
    test_cache = numpy.concatenate((test_cache, numpy.array([y_test]).T), axis=1)
    return X_train_feature, train_cache, X_validate_feature, validate_cache, X_test_feature, test_cache


def sample_copy_stop_condition(workflow_type: int, process_num: int, pos_num: int):
    """
    the condition to stop getting samples for training:
        when sample_size >= 9000 and pos_num >= 2000 and neg_num >= 2000, stop
    """
    if workflow_type == TWITTER_TYPE:
        # if process_num >= 10000 and pos_num >= 2000 and (process_num - pos_num) >= 2000:
        if process_num >= TWITTER_SAMPLE_COPY_SIZE and pos_num >= TWITTER_SAMPLE_COPY_SIZE / 10 and (
                process_num - pos_num) >= TWITTER_SAMPLE_COPY_SIZE / 10:
            # if process_num >= 10000 and pos_num >= 1000 and (process_num - pos_num) >= 1000:
            return True
        else:
            return False
    elif workflow_type == COCO_TYPE:
        if process_num >= COCO_SAMPLE_COPY_SIZE and pos_num >= COCO_SAMPLE_COPY_SIZE / 10 and (
                process_num - pos_num) >= COCO_SAMPLE_COPY_SIZE / 10:
            return True
        else:
            return False
    else:
        if process_num >= UCF101_SAMPLE_COPY_SIZE and pos_num >= UCF101_SAMPLE_COPY_SIZE / 10 and (
                process_num - pos_num) >= UCF101_SAMPLE_COPY_SIZE / 10:
            return True
        else:
            return False


def copy_samples(workflow_type: int, sample: List[Record]):
    """
    if the sample doesn't satisfy the stop_condition, copy the sample.
    :param workflow_type: the type of workflow, different workflow_type using different sample_train_stop_condition
    :param sample: input sample
    :return: a sample satisfy the stop_condition
    """
    process_num = len(sample)
    pass_num = 0
    for record in sample:
        if record["flag"] == 1:
            pass_num += 1
    labeled_sample_flag = sample_copy_stop_condition(workflow_type=workflow_type, process_num=process_num,
                                                     pos_num=pass_num)
    if labeled_sample_flag:
        print("\tlabeled sample copy num = " + str(len(sample)) + "\tpass_num = " + str(pass_num))
        return sample
    else:
        sample_copy = sample.copy()
        index = 0
        while not labeled_sample_flag:
            record = sample[index]
            if record["flag"] == 1:
                pass_num += 1
            process_num += 1
            sample_copy.append(sample[index])
            labeled_sample_flag = sample_copy_stop_condition(workflow_type=workflow_type, process_num=process_num,
                                                             pos_num=pass_num)
            index += 1
            if index == len(sample):
                index = 0
        print("\tlabeled sample copy num = " + str(len(sample_copy)) + "\tpass_num = " + str(pass_num))
        return sample_copy


def train_validate_test_split(sample: List[Record]):
    """
    randomly split a labeled sample dataset into training, validation, and testing dataset
    """
    random.seed(1)
    random.shuffle(sample)
    for record in sample[:int(len(sample) * 0.6)]:
        record["split_flag"] = 0
    for record in sample[int(len(sample) * 0.6):int(len(sample) * 0.8)]:
        record["split_flag"] = 1
    for record in sample[int(len(sample) * 0.8):]:
        record["split_flag"] = 2


def retrain_pp(workflow_pp_type: int, X_train_feature, train_cache, X_validate_feature, validate_cache, X_test_feature,
               test_cache):
    """
    retrain a pp
    :return: a retrained pp
    """
    if workflow_pp_type == TWITTER_PP_TYPE:
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
    return proba_predicator


def get_selectivity(samples: List[Record]):
    """
    get a samples selectivity
    """
    process_num, pass_num = len(samples), 0
    for record in samples:
        if record["flag"] == 1:
            pass_num += 1
    return 0 if process_num == 0 else pass_num / process_num


def batch_records_copy(batch_records: List[Record]) -> List[Record]:
    """
    copy a list of records
    """
    copy_records = []
    for record in batch_records:
        new_record = record.copy()
        copy_records.append(new_record)
    return copy_records
