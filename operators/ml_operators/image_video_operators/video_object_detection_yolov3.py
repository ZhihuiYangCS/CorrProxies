import os
import time
from typing import Any, List
from collections import Counter
import rootpath

rootpath.append()
from paths import YOLO_DIR, YOLO_WEIGHTS_PATH, VIDEO_UCF101_TRAIN_DATA_PATH, UCF101_FRAMES_PATH
import sys

sys.path.insert(0, YOLO_DIR)
from operators.operator_base.operator_parallel import OperatorParallel
from my_image_detector import load_model, detect
from operators.scan.video_json_scan import VideoJsonScan
from records.record import Record
from operators.operator_base.operator_utility import BatchOutput


class VideoObjectDetectionYolov3(OperatorParallel):
    """
    https://github.com/ultralytics/yolov3
    https://pjreddie.com/darknet/yolo/
    This class implments the yolov3 object detection operator
    """

    def __init__(self):
        super(VideoObjectDetectionYolov3, self).__init__()

    def get_operator_cost(self):
        """return operator time cost per record, in ms/record
            :raises ValueError
        """
        if VideoObjectDetectionYolov3.operator_cost is None:
            if self.process_count == 0:
                print("Getting operator cost: process_count = " + str(
                    self.process_count) + "\tYou may forget add processing statistic informations ...")
                VideoObjectDetectionYolov3.operator_cost = None
            else:
                VideoObjectDetectionYolov3.operator_cost = self.process_time / self.process_count

    def load_model(self) -> Any:
        time1 = time.time()
        model = load_model(weights_path=YOLO_WEIGHTS_PATH)
        print(self.operator_name + ": finish loading model. Loading model cost = " + str(time.time() - time1))
        return model

    def _process(self, one: Record, model=None) -> Any:
        two_paths = os.path.split(one.value)
        frames_path = os.path.join(UCF101_FRAMES_PATH, two_paths[0], two_paths[1].split(".")[0])
        frames_list = os.listdir(frames_path)
        index_1, index_2 = int(len(frames_list) / 2) - 1, int(len(frames_list) / 2)
        results = detect(path_list=[os.path.join(frames_path, frames_list[index_1]),
                                    os.path.join(frames_path, frames_list[index_2])], model=model)
        returned_result = Counter(dict(results[0])) + Counter(dict(results[1]))
        return dict(returned_result)

    def process_batch_inner(self, batch: List[Record], model=None) -> BatchOutput:
        processed_batch, unprocessed_batch_index, unprocessed_batch = {}, [], []
        for index, record in enumerate(batch):
            if self.get_result(record) is None:
                unprocessed_batch_index.append(index)
                unprocessed_batch.append(record)
            else:
                processed_batch[index] = record
        if unprocessed_batch:
            start_time = time.time()
            path_list = []
            for record in unprocessed_batch:
                two_paths = os.path.split(record.value)
                frames_path = os.path.join(UCF101_FRAMES_PATH, two_paths[0], two_paths[1].split(".")[0])
                frames_list = os.listdir(frames_path)
                index_1, index_2 = int(len(frames_list) / 2) - 1, int(len(frames_list) / 2)
                path_list.append(os.path.join(frames_path, frames_list[index_1]))
                path_list.append(os.path.join(frames_path, frames_list[index_2]))
            results = detect(path_list=path_list, model=model)
            for index, record in enumerate(unprocessed_batch):
                record_result = Counter(dict(results[index*2])) + Counter(dict(results[index*2+1]))
                self.set_result(one=record, result=dict(record_result))
            end_time = time.time()
            if self.operator_cost is None:
                processed_time = end_time - start_time
            else:
                processed_time = self.operator_cost * len(unprocessed_batch)
            processed_count = len(unprocessed_batch)
            passed_count = len(unprocessed_batch)
            for i in range(len(unprocessed_batch_index)):
                processed_batch[unprocessed_batch_index[i]] = unprocessed_batch[i]
        else:
            processed_count = 0
            passed_count = 0
            processed_time = 0
        returned_batch = [processed_batch[key] for key in sorted(processed_batch.keys())]
        return BatchOutput(returned_batch=returned_batch, processed_count=processed_count, passed_count=passed_count,
                           processed_time=processed_time)

    def get_new_object(self, previous_operators=None, operator_name: str = ""):
        """
        return a new object
        :param previous_operators: List[Operator]
        :param operator_name: str, default value is ""
        """
        return VideoObjectDetectionYolov3()


"""
test units
"""


def operator_name_test():
    video_object_detection = VideoObjectDetectionYolov3()
    video_object_detection_model = video_object_detection.load_model()
    print(video_object_detection)


def operator_test():
    """
    operator block process test
    :return:
    """
    scan = VideoJsonScan(VIDEO_UCF101_TRAIN_DATA_PATH)
    block_records = scan.get_block_records(10)
    operator = VideoObjectDetectionYolov3()
    model = operator.load_model()
    batch_output = operator.process_batch(batch=block_records, model=model)
    for record in batch_output.returned_batch:
        print(str(operator.get_result(record)).encode('utf-8'))
    print(batch_output)
    scan.close()


def operator_copy_test():
    time0 = time.time()
    operator = VideoObjectDetectionYolov3()
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
    operator = VideoObjectDetectionYolov3()
    model = operator.load_model()
    batch_output = operator.process_batch(batch=block_records, model=model)
    operator.get_operator_cost()
    print("operator cost = " + str(operator.operator_cost) + "\t " + str(batch_output) + "\n")
    time2 = time.time()

    block_records = scan.get_block_records(100)
    batch_output = operator.process_batch(batch=block_records, model=model)
    operator.get_operator_cost()
    print("operator cost = " + str(operator.operator_cost) + "\t " + str(batch_output) + "\n")

    operator_new = VideoObjectDetectionYolov3()
    print("new operator cost = " + str(operator_new.operator_cost) + "\n")

    scan.close()
    time3 = time.time()
    print("one process: one block time = " + str(time2 - time1) + "\t two blocks time = " + str(time3 - time2) + "\n\n")


def multple_processes_test():
    time1 = time.time()
    scan = VideoJsonScan(VIDEO_UCF101_TRAIN_DATA_PATH)
    block_records = scan.get_block_records(100)
    operator = VideoObjectDetectionYolov3()
    batch_output = operator.multiple_processes_batch(batch=block_records, num_process=2)
    operator.get_operator_cost()
    print(batch_output.returned_batch[0])
    print("operator cost = " + str(operator.operator_cost) + "\t " + str(batch_output) + "\n")
    time2 = time.time()

    block_records = scan.get_block_records(100)
    batch_output = operator.multiple_processes_batch(batch=block_records, num_process=2)
    operator.get_operator_cost()
    print(batch_output.returned_batch[0])
    print("operator cost = " + str(operator.operator_cost) + "\t " + str(batch_output) + "\n")

    operator_new = VideoObjectDetectionYolov3()
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
