import gc
import os
import time
import warnings
from typing import Any, List
import numpy
warnings.filterwarnings('ignore')
import rootpath

rootpath.append()
from model.keras_video_classifier import VGG16BidirectionalLSTMVideoClassifier
from operators.operator_base.operator_utility import BatchOutput
from operators.operator_base.operator_parallel import OperatorParallel
from operators.scan.video_json_scan import VideoJsonScan
from paths import KERAS_MODEL_PATH, VIDEO_UCF101_TRAIN_DATA_PATH, UCF101_VIDEO_PATH
from records.record import Record


class VideoActivityRecognition(OperatorParallel):
    """
    https://github.com/chen0040/keras-video-classifier
    This class implements the keras_video_classifier used for video activity recognition
    """

    def __init__(self):
        super(VideoActivityRecognition, self).__init__()
        vgg16_include_top = True
        self.config_file_path = VGG16BidirectionalLSTMVideoClassifier.get_config_file_path(KERAS_MODEL_PATH, vgg16_include_top=vgg16_include_top)
        self.weight_file_path = VGG16BidirectionalLSTMVideoClassifier.get_weight_file_path(KERAS_MODEL_PATH, vgg16_include_top=vgg16_include_top)

    def get_operator_cost(self):
        """return operator time cost per record, in ms/record
            :raises ValueError
        """
        if VideoActivityRecognition.operator_cost is None:
            if self.process_count == 0:
                print("Getting operator cost: process_count = " + str(
                    self.process_count) + "\tYou may forget add processing statistic informations ...")
                VideoActivityRecognition.operator_cost = None
            else:
                VideoActivityRecognition.operator_cost = self.process_time / self.process_count

    def load_model(self) -> Any:
        time1 = time.time()
        numpy.random.seed(42)
        predictor = VGG16BidirectionalLSTMVideoClassifier()
        predictor.load_model(self.config_file_path, self.weight_file_path)
        print(self.operator_name + ": finish loading model. Loading model cost = " + str(time.time() - time1))
        return predictor

    def _process(self, one: Record, model=None) -> Any:
        predict_label = model.predict(os.path.join(UCF101_VIDEO_PATH, one.value))
        return predict_label

    def multiple_processes_batch(self, batch: List[Record], num_process: int) -> BatchOutput:
        if num_process == 1:
            num_process = num_process
        else:
            num_process = num_process * 3
        gc.collect()
        return self.multiple_processes_batch_inner(batch=batch, num_process=num_process)

    def get_new_object(self, previous_operators=None, operator_name: str = ""):
        """
        return a new object
        :param previous_operators: List[Operator]
        :param operator_name: str, default value is ""
        """
        return VideoActivityRecognition()


"""
test units
"""


def operator_name_test():
    video_activity_recognition = VideoActivityRecognition()
    video_activity_recognition_model = video_activity_recognition.load_model()
    print(video_activity_recognition)


def operator_test():
    """
    operator block process test
    :return:
    """
    scan = VideoJsonScan(VIDEO_UCF101_TRAIN_DATA_PATH)
    block_records = scan.get_block_records(10)
    operator = VideoActivityRecognition()
    model = operator.load_model()
    batch_output = operator.process_batch(batch=block_records, model=model)
    for record in batch_output.returned_batch:
        print(str(operator.get_result(record)).encode('utf-8'))
    print(batch_output)
    scan.close()


def operator_copy_test():
    time0 = time.time()
    operator = VideoActivityRecognition()
    time1 = time.time()
    operator_copy = operator.get_new_object()
    print("create model time = " + str(time1 - time0) + "\tcopy model time = " + str(time.time() - time1))

    print("operator.process_count = " + str(operator.process_count))
    print("operator_copy.process_count = " + str(operator_copy.process_count))

    operator.process_count = 1000
    print("operator.process_count = " + str(operator.process_count))
    print("operator_copy.process_count = " + str(operator_copy.process_count))
    operator_copy.process_count = 2000
    print("operator.process_count = " + str(operator.process_count))
    print("operator_copy.process_count = " + str(operator_copy.process_count))
    operator.operator_reset()
    print("operator.process_count = " + str(operator.process_count))
    print("operator_copy.process_count = " + str(operator_copy.process_count))
    operator_copy.operator_reset()
    print("operator.process_count = " + str(operator.process_count))
    print("operator_copy.process_count = " + str(operator_copy.process_count))


def operator_cost_test():
    time1 = time.time()
    scan = VideoJsonScan(VIDEO_UCF101_TRAIN_DATA_PATH)
    block_records = scan.get_block_records(100)
    operator = VideoActivityRecognition()
    model = operator.load_model()
    batch_output = operator.process_batch(batch=block_records, model=model)
    operator.get_operator_cost()
    print("operator cost = " + str(operator.operator_cost) + "\t " + str(batch_output) + "\n")
    time2 = time.time()

    block_records = scan.get_block_records(100)
    batch_output = operator.process_batch(batch=block_records, model=model)
    operator.get_operator_cost()
    print("operator cost = " + str(operator.operator_cost) + "\t " + str(batch_output) + "\n")

    operator_new = VideoActivityRecognition()
    print("new operator cost = " + str(operator_new.operator_cost) + "\n")

    scan.close()
    time3 = time.time()
    print("one process: one block time = " + str(time2 - time1) + "\t two blocks time = " + str(time3 - time2) + "\n\n")


def multple_processes_test():
    time1 = time.time()
    scan = VideoJsonScan(VIDEO_UCF101_TRAIN_DATA_PATH)
    block_records = scan.get_block_records(100)
    operator = VideoActivityRecognition()
    batch_output = operator.multiple_processes_batch(batch=block_records, num_process=2)
    operator.get_operator_cost()
    print("operator cost = " + str(operator.operator_cost) + "\t " + str(batch_output) + "\n")
    time2 = time.time()

    block_records = scan.get_block_records(100)
    batch_output = operator.multiple_processes_batch(batch=block_records, num_process=2)
    operator.get_operator_cost()
    print("operator cost = " + str(operator.operator_cost) + "\t " + str(batch_output) + "\n")

    operator_new = VideoActivityRecognition()
    print("new operator cost = " + str(operator_new.operator_cost) + "\n")

    scan.close()
    time3 = time.time()
    print("multiple process: one block time = " + str(time2 - time1) + "\t two blocks time = " + str(
        time3 - time2) + "\n")


if __name__ == '__main__':
    # operator_name_test()
    # operator_test()
    # operator_copy_test()
    # operator_cost_test()
    multple_processes_test()
