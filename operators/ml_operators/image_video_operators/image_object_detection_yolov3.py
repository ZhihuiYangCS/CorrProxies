import os
import time
from typing import Any, List
import rootpath

rootpath.append()
from paths import YOLO_DIR
import sys

sys.path.insert(0, YOLO_DIR)
from my_image_detector import load_model, detect
from operators.scan.image_json_scan import ImageJsonScan
from paths import YOLO_WEIGHTS_PATH, IMAGE_COCO_TRAIN_DATA_PATH, COCO_IMAGE_PATH
from records.record import Record
from operators.operator_base.operator_parallel import OperatorParallel
from operators.operator_base.operator_utility import BatchOutput


class ImageObjectDetectionYolov3(OperatorParallel):
    """
    git clone https://github.com/ultralytics/yolov3
    https://pjreddie.com/darknet/yolo/
    This class implments the yolov3 object detection operator
    """

    def __init__(self):
        super(ImageObjectDetectionYolov3, self).__init__()

    def load_model(self) -> Any:
        time1 = time.time()
        model = load_model(weights_path=YOLO_WEIGHTS_PATH)
        print(self.operator_name + ": finish loading model. Loading model cost = " + str(time.time() - time1))
        return model

    def _process(self, one: Record, model=None) -> Any:
        result = detect(path_list=[os.path.join(COCO_IMAGE_PATH, one.value)], model=model)
        if result is not None:
            return dict(result[0])
        else:
            return None

    def get_operator_cost(self):
        """return operator time cost per record, in ms/record
            :raises ValueError
        """
        if ImageObjectDetectionYolov3.operator_cost is None:
            if self.process_count == 0:
                print("Getting operator cost: process_count = " + str(
                    self.process_count) + "\tYou may forget add processing statistic informations ...")
                ImageObjectDetectionYolov3.operator_cost = None
            else:
                ImageObjectDetectionYolov3.operator_cost = self.process_time / self.process_count

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
            path_list = [os.path.join(COCO_IMAGE_PATH, record.value) for record in unprocessed_batch]
            results = detect(path_list=path_list, model=model)
            for index, record in enumerate(unprocessed_batch):
                self.set_result(one=record, result=results[index])
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
        return ImageObjectDetectionYolov3()


"""
test units
"""


def operator_name_test():
    image_object_detection = ImageObjectDetectionYolov3()
    image_object_detection_model = image_object_detection.load_model()
    print(image_object_detection)


def operator_test():
    """
    operator block process test
        b"{'car': 0.9698144793510437, 'kite': 0.5037378668785095, 'person': 0.3092358708381653, 'chair': 0.33232274651527405}"
        b"{'motorcycle': 0.34602513909339905, 'person': 0.31571072340011597}"
        b"{'person': 0.4356561601161957, 'car': 0.8537365794181824, 'traffic light': 0.3192957937717438, 'truck': 0.35007211565971375}"
        b"{'dining table': 0.9885171055793762, 'apple': 0.7136256098747253, 'knife': 0.7085263133049011, 'fork': 0.470584511756897}"
        b"{'person': 0.3238706588745117, 'train': 0.5895283222198486}"
        b"{'person': 0.9985597133636475, 'horse': 0.8955840468406677}"
        b"{'person': 0.30903762578964233, 'skis': 0.5730937719345093, 'snowboard': 0.5516861081123352}"
        b"{'kite': 0.9633829593658447}"
        b"{'person': 0.4595141112804413, 'car': 0.981898307800293, 'handbag': 0.8456006050109863}"
        b"{'horse': 0.3560161888599396, 'person': 0.3789132833480835, 'car': 0.9520999193191528, 'bus': 0.9306688904762268, 'traffic light': 0.4787653088569641}"
    :return:
    """
    scan = ImageJsonScan(IMAGE_COCO_TRAIN_DATA_PATH)
    block_records = scan.get_block_records(10)
    operator = ImageObjectDetectionYolov3()
    model = operator.load_model()
    batch_output = operator.process_batch(batch=block_records, model=model)
    for record in batch_output.returned_batch:
        print(str(operator.get_result(record)).encode('utf-8'))
    print(batch_output)
    scan.close()


def operator_copy_test():
    time0 = time.time()
    operator = ImageObjectDetectionYolov3()
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
    scan = ImageJsonScan(IMAGE_COCO_TRAIN_DATA_PATH)
    block_records = scan.get_block_records(100)
    operator = ImageObjectDetectionYolov3()
    model = operator.load_model()
    batch_output = operator.process_batch(batch=block_records, model=model)
    operator.get_operator_cost()
    print("operator cost = " + str(operator.operator_cost) + "\t " + str(batch_output) + "\n")
    time2 = time.time()

    block_records = scan.get_block_records(100)
    batch_output = operator.process_batch(batch=block_records, model=model)
    operator.get_operator_cost()
    print("operator cost = " + str(operator.operator_cost) + "\t " + str(batch_output) + "\n")

    operator_new = ImageObjectDetectionYolov3()
    print("new operator cost = " + str(operator_new.operator_cost) + "\n")

    scan.close()
    time3 = time.time()
    print("one process: one block time = " + str(time2 - time1) + "\t two blocks time = " + str(time3 - time2) + "\n\n")


def multple_processes_test():
    time1 = time.time()
    scan = ImageJsonScan(IMAGE_COCO_TRAIN_DATA_PATH)
    block_records = scan.get_block_records(100)
    operator = ImageObjectDetectionYolov3()
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

    operator_new = ImageObjectDetectionYolov3()
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
