import os
import time
from typing import Any

import numpy
import rootpath

rootpath.append()
from operators.scan.video_json_scan import VideoJsonScan
from paths import VIDEO_UCF101_TRAIN_DATA_PATH, UCF101_FRAMES_PATH
from optimizer.PPs.preprocessor.preprocessor import Preprocessor
from model.image_features import init_model, predict_image_features
from records.record import Record


class VideoPreprocessor(Preprocessor):
    """
    This operator preprocess images by converting image to 'numpy.ndarray'
    """

    def __init__(self):
        super(VideoPreprocessor, self).__init__()

    def get_operator_cost(self):
        """return operator time cost per record, in ms/record
            :raises ValueError
        """
        if VideoPreprocessor.operator_cost is None:
            if self.process_count == 0:
                print("Getting operator cost: process_count = " + str(
                    self.process_count) + "\tYou may forget add processing statistic informations ...")
                VideoPreprocessor.operator_cost = None
            else:
                VideoPreprocessor.operator_cost = self.process_time / self.process_count

    def load_model(self) -> Any:
        """
        https://github.com/chsasank/image_features
        It supports many models. For exmaple: alexnet, densenet121, densenet169, densenet201, densenet161
        resnet18, resnet34, resnet50, resnet101, resnet152,
        vgg11, vgg11_bn, vgg13, vgg13_bn, vgg16, vgg16_bn, vgg19_bn, vgg19
        """
        time1 = time.time()
        model = init_model(model_name='resnet18')
        print(self.operator_name + ": finish loading model. Loading model cost = " + str(time.time() - time1))
        return model

    def _process(self, one: Record, model=None) -> Any:
        two_paths = os.path.split(one.value)
        frames_path = os.path.join(UCF101_FRAMES_PATH, two_paths[0], two_paths[1].split(".")[0])
        frames_list = os.listdir(frames_path)
        if len(frames_list) == 1:
            result = predict_image_features([os.path.join(frames_path, frames_list[0])], model)
            result = numpy.hstack((result, result))
        elif len(frames_list) == 2:
            result = predict_image_features([os.path.join(frames_path, frames_list[0])], model)
            result = numpy.hstack(
                (result, predict_image_features([os.path.join(frames_path, frames_list[1])], model)))
        else:
            index_1, index_2 = int(len(frames_list) / 2) - 1, int(len(frames_list) / 2)
            result = predict_image_features([os.path.join(frames_path, frames_list[index_1])], model)
            result = numpy.hstack(
                (result, predict_image_features([os.path.join(frames_path, frames_list[index_2])], model)))
        return result

    def get_new_object(self, previous_operators=None, operator_name: str = ""):
        """
        return a new object
        :param previous_operators: List[Operator]
        :param operator_name: str, default value is ""
        """
        return VideoPreprocessor()


"""
test units
"""


def operator_name_test():
    video_preprocessor = VideoPreprocessor()
    video_preprocessor.load_model()
    print(video_preprocessor)


def operator_test():
    """
    video_preprocessor block process test
    :return:
    """
    scan = VideoJsonScan(VIDEO_UCF101_TRAIN_DATA_PATH)
    block_records = scan.get_block_records(100)
    operator = VideoPreprocessor()
    model = operator.load_model()
    batch_output = operator.process_batch(batch=block_records, model=model)
    for record in batch_output.returned_batch:
        print(str(operator.get_result(record)).encode('utf-8'))
    operator.get_operator_cost()
    operator.print_operator_detail()
    print(batch_output)
    print(batch_output.returned_batch[0])
    scan.close()


def operator_copy_test():
    time0 = time.time()
    operator = VideoPreprocessor()
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
    operator = VideoPreprocessor()
    model = operator.load_model()
    batch_output = operator.process_batch(batch=block_records, model=model)
    operator.get_operator_cost()
    operator.print_operator_detail()
    print(batch_output)
    print(batch_output.returned_batch[0])
    print("\n")
    time2 = time.time()

    block_records = scan.get_block_records(100)
    batch_output = operator.process_batch(batch=block_records, model=model)
    operator.get_operator_cost()
    operator.print_operator_detail()
    print(batch_output)
    print(batch_output.returned_batch[0])
    print("\n")

    operator_new = VideoPreprocessor()
    print("new operator cost = " + str(operator_new.operator_cost) + "\n")

    scan.close()
    time3 = time.time()
    print("one process: one block time = " + str(time2 - time1) + "\t two blocks time = " + str(time3 - time2) + "\n\n")


def multple_threads_test():
    time1 = time.time()
    scan = VideoJsonScan(VIDEO_UCF101_TRAIN_DATA_PATH)
    block_records = scan.get_block_records(100)
    operator = VideoPreprocessor()
    model = operator.load_model()
    batch_output = operator.multiple_threads_batch(batch=block_records, num_process=2, model=model)
    operator.get_operator_cost()
    operator.print_operator_detail()
    print(batch_output)
    print(batch_output.returned_batch[0])
    print("\n")
    time2 = time.time()

    block_records = scan.get_block_records(100)
    batch_output = operator.multiple_threads_batch(batch=block_records, num_process=2, model=model)
    operator.get_operator_cost()
    operator.print_operator_detail()
    print(batch_output)
    print(batch_output.returned_batch[0])
    print("\n")

    operator_new = VideoPreprocessor()
    print("new operator cost = " + str(operator_new.operator_cost) + "\n")

    scan.close()
    time3 = time.time()
    print("multiple process: one block time = " + str(time2 - time1) + "\t two blocks time = " + str(
        time3 - time2) + "\n")


def multple_processes_test():
    time1 = time.time()
    scan = VideoJsonScan(VIDEO_UCF101_TRAIN_DATA_PATH)
    block_records = scan.get_block_records(100)
    operator = VideoPreprocessor()
    batch_output = operator.multiple_processes_batch(batch=block_records, num_process=2)
    operator.get_operator_cost()
    operator.print_operator_detail()
    print(batch_output)
    print(batch_output.returned_batch[0])
    print("\n")
    time2 = time.time()

    block_records = scan.get_block_records(100)
    batch_output = operator.multiple_processes_batch(batch=block_records, num_process=2)
    operator.get_operator_cost()
    operator.print_operator_detail()
    print(batch_output)
    print(batch_output.returned_batch[0])
    print("\n")

    operator_new = VideoPreprocessor()
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
    # multple_threads_test()
    multple_processes_test()
