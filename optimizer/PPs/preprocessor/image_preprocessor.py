import os
import time
from typing import Any, List

import rootpath


rootpath.append()
from operators.operator_base.operator_utility import BatchOutput
from optimizer.PPs.preprocessor.preprocessor import Preprocessor
from model.image_features import init_model, predict_image_features
from operators.scan.image_json_scan import ImageJsonScan
from paths import IMAGE_COCO_TRAIN_DATA_PATH, COCO_IMAGE_PATH
from records.record import Record


class ImagePreprocessor(Preprocessor):
    """
    This operator preprocess images by converting image to 'numpy.ndarray'
    """

    def __init__(self):
        super(ImagePreprocessor, self).__init__()

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
        result = predict_image_features([os.path.join(COCO_IMAGE_PATH, one.value)], model)
        return result

    def get_operator_cost(self):
        """return operator time cost per record, in ms/record
            :raises ValueError
        """
        if ImagePreprocessor.operator_cost is None:
            if self.process_count == 0:
                print("Getting operator cost: process_count = " + str(
                    self.process_count) + "\tYou may forget add processing statistic informations ...")
                ImagePreprocessor.operator_cost = None
            else:
                ImagePreprocessor.operator_cost = self.process_time / self.process_count

    def get_new_object(self, previous_operators=None, operator_name: str = ""):
        """
        return a new object
        :param previous_operators: List[Operator]
        :param operator_name: str, default value is ""
        """
        return ImagePreprocessor()


"""
test units
"""


def operator_name_test():
    image_preprocessor = ImagePreprocessor()
    image_preprocessor.load_model()
    print(image_preprocessor)


def operator_test():
    """
    image_preprocessor block process test
    :return:
    """
    scan = ImageJsonScan(IMAGE_COCO_TRAIN_DATA_PATH)
    block_records = scan.get_block_records(10)
    operator = ImagePreprocessor()
    model = operator.load_model()
    batch_output = operator.process_batch(batch=block_records, model=model)
    for record in batch_output.returned_batch:
        print(str(operator.get_result(record)).encode('utf-8'))
    operator.get_operator_cost()
    print(operator)
    print(batch_output)
    print(batch_output.returned_batch[0])
    scan.close()


def operator_copy_test():
    time0 = time.time()
    operator = ImagePreprocessor()
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
    operator = ImagePreprocessor()
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

    operator_new = ImagePreprocessor()
    print("new operator cost = " + str(operator_new.operator_cost) + "\n")

    scan.close()
    time3 = time.time()
    print("one process: one block time = " + str(time2 - time1) + "\t two blocks time = " + str(time3 - time2) + "\n\n")


def multple_threads_test():
    time1 = time.time()
    scan = ImageJsonScan(IMAGE_COCO_TRAIN_DATA_PATH)
    block_records = scan.get_block_records(100)
    operator = ImagePreprocessor()
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

    operator_new = ImagePreprocessor()
    print("new operator cost = " + str(operator_new.operator_cost) + "\n")

    scan.close()
    time3 = time.time()
    print("multiple process: one block time = " + str(time2 - time1) + "\t two blocks time = " + str(
        time3 - time2) + "\n")


def multple_processes_test():
    time1 = time.time()
    scan = ImageJsonScan(IMAGE_COCO_TRAIN_DATA_PATH)
    block_records = scan.get_block_records(100)
    operator = ImagePreprocessor()
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

    operator_new = ImagePreprocessor()
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
