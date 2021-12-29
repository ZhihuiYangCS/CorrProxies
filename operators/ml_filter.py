import time
from typing import Any, List

import rootpath


rootpath.append()
from operators.ml_operators.text_operators.spacy.spacy_operator import POSTaggerSpacySM
from operators.ml_operators.image_video_operators.image_object_detection_yolov3 import ImageObjectDetectionYolov3
from operators.scan.image_json_scan import ImageJsonScan
from operators.operator_base.operator_action import OperatorDynamicAction
from operators.ml_predicator import MLPredicator
from operators.scan.text_json_scan import TextJsonScan
from paths import TEXT_TWITTER_TRAIN_DATA_PATH, IMAGE_COCO_TRAIN_DATA_PATH
from records.record import Record
from operators.operator_base.operator_utility import BatchOutput


class MLFilter(OperatorDynamicAction):
    """
    a filter operator drops out records based on a ml_predicator
    """

    def __init__(self, ml_predicator: MLPredicator):
        """
        pass in a ml_predicator, and calculate the pass rate of the filter
        """
        super().__init__(ml_predicator.operator_name)
        self.ml_predicator = ml_predicator

    def get_operator_cost(self):
        """return operator time cost per record, in ms/record
            :raises ValueError
        """
        self.ml_predicator.get_operator_cost()
        if self.operator_cost is None:
            if self.process_count == 0:
                print("Getting operator cost: process_count = " + str(
                    self.process_count) + "\tYou may forget add processing statistic informations ...")
                self.operator_cost = None
            else:
                self.operator_cost = self.process_time / self.process_count

    def load_model(self) -> Any:
        return self.ml_predicator.ml_operator.load_model()

    def _process(self, one: Record, model=None) -> Any:
        pass

    def filter_process(self, batch: List[Record], num_process: int = 0, model: Any = None,
                       mode: int = 0) -> BatchOutput:
        """
        :param batch:List[Record] a list of records to be processed
        :param num_process: int the number of processes or threads when mode is multiple_threads_batch or multiple_processes_batch
        :param model: Any an ML model, it is used in process_one, process_batch, and multiple_threads_batch
        :param mode:int 0 -> process_one
                        1 -> process_batch
                        2 -> multiple_threads_batch
                        3 -> multiple_processes_batch
        """
        returned_batch, processed_count, passed_count, processed_time = [], 0, 0, 0
        if mode == 0:
            ml_operator_batch_output = self.ml_predicator.process_one(record=batch[0], model=model)
        elif mode == 1:
            ml_operator_batch_output = self.ml_predicator.process_batch(batch=batch, model=model)
        elif mode == 2:
            ml_operator_batch_output = self.ml_predicator.multiple_threads_batch(batch=batch, num_process=num_process,
                                                                                 model=model)
        else:
            ml_operator_batch_output = self.ml_predicator.multiple_processes_batch(batch=batch, num_process=num_process)
        start_time = time.time()
        for record in ml_operator_batch_output.returned_batch:
            processed_count += 1
            label = self.ml_predicator.get_result(one=record)
            if label:
                returned_batch.append(record)
                passed_count += 1
        end_time = time.time()
        if self.operator_cost is None:
            processed_time = ml_operator_batch_output.processed_time + (end_time - start_time)
        else:
            processed_time = self.operator_cost * processed_count
        returned_batch_output = BatchOutput(returned_batch=returned_batch, processed_count=processed_count,
                                            passed_count=passed_count, processed_time=processed_time)
        self.set_processed_statistics(batch_output=returned_batch_output)
        return returned_batch_output

    def process_one(self, record: Record, model=None) -> BatchOutput:
        """
        filter or not
        """
        return self.filter_process(batch=[record], model=model, mode=0)

    def process_batch(self, batch: List[Record], model=None) -> BatchOutput:
        return self.filter_process(batch=batch, model=model, mode=1)

    def multiple_threads_batch(self, batch: List[Record], num_process: int, model=None) -> BatchOutput:
        return self.filter_process(batch=batch, num_process=num_process, model=model, mode=2)

    def multiple_processes_batch(self, batch: List[Record], num_process: int) -> BatchOutput:
        return self.filter_process(batch=batch, num_process=num_process, mode=3)

    def operator_reset(self):
        """
        reset the operator status to initial state.
        """
        self.process_count = 0
        self.process_time = 0
        self.pass_count = 0
        self.ml_predicator.operator_reset()

    def get_new_object(self, previous_operators=None, operator_name: str = ""):
        """
        return a new object
        :param previous_operators: List[Operator]
        :param operator_name: str, default value is ""
        """
        ml_predicate = self.ml_predicator.get_new_object(previous_operators=previous_operators,
                                                         operator_name=operator_name)
        new_filter = MLFilter(ml_predicator=ml_predicate)
        if self.operator_cost is not None:
            new_filter.operator_cost = self.operator_cost
        return new_filter


"""
test unit
"""


def ml_filter_test():
    """
    the ml_filter
        test ml_filter's pass_num, process_num, pass_rate, cost
        test ml_operator's cost
    """
    scan = TextJsonScan(TEXT_TWITTER_TRAIN_DATA_PATH)
    block_records = scan.get_block_records(100)
    pos_tagger = POSTaggerSpacySM()
    time1 = time.time()

    def f(result: Any):
        return any([value[0] in ['VBD', 'WRB', 'PRP'] for token, value in result.items()])

    ml_predicate1 = MLPredicator(ml_operator=pos_tagger, predicate=f)
    ml_filter_1 = MLFilter(ml_predicator=ml_predicate1)

    batch_output1 = ml_filter_1.process_batch(batch=block_records, model=ml_filter_1.load_model())
    ml_filter_1.get_operator_cost()
    ml_filter_1.print_operator_detail()
    ml_filter_1.ml_predicator.print_operator_detail()
    ml_filter_1.ml_predicator.ml_operator.print_operator_detail()
    print("the number of records after " + ml_filter_1.operator_name + " = " + str(len(batch_output1.returned_batch)))
    time2 = time.time()

    def g(result: Any):
        return any([value[0] in ['VBD', 'WRB', 'PRP'] for token, value in result.items()])

    ml_predicate2 = MLPredicator(ml_operator=pos_tagger, predicate=g)
    ml_filter_2 = MLFilter(ml_predicator=ml_predicate2)
    batch_output2 = ml_filter_2.process_batch(block_records, model=ml_filter_2.load_model())
    ml_filter_2.get_operator_cost()
    ml_filter_2.print_operator_detail()
    ml_filter_2.ml_predicator.print_operator_detail()
    ml_filter_2.ml_predicator.ml_operator.print_operator_detail()
    print("the number of records after " + ml_filter_2.operator_name + " = " + str(len(batch_output2.returned_batch)))
    time3 = time.time()
    scan.close()
    print("first predict time = " + str(time2 - time1) + "\t second predict time = " + str(time3 - time2) + "\n")


def multithreading_test():
    """
    the ml_filter
        test ml_filter's pass_num, process_num, pass_rate, cost
        test ml_operator's cost
    """
    scan = TextJsonScan(TEXT_TWITTER_TRAIN_DATA_PATH)
    block_records = scan.get_block_records(100)
    pos_tagger = POSTaggerSpacySM()
    time1 = time.time()

    def f(result: Any):
        return any([value[0] in ['VBD', 'WRB', 'PRP'] for token, value in result.items()])

    ml_predicate1 = MLPredicator(ml_operator=pos_tagger, predicate=f)
    ml_filter_1 = MLFilter(ml_predicator=ml_predicate1)

    batch_output1 = ml_filter_1.multiple_threads_batch(batch=block_records, num_process=10,
                                                       model=ml_filter_1.load_model())
    ml_filter_1.get_operator_cost()
    ml_filter_1.print_operator_detail()
    ml_filter_1.ml_predicator.print_operator_detail()
    ml_filter_1.ml_predicator.ml_operator.print_operator_detail()
    print("the number of records after " + ml_filter_1.operator_name + " = " + str(len(batch_output1.returned_batch)))
    scan.close()
    time2 = time.time()
    print("multithreading cost = " + str(time2 - time1) + "\n")


def multiprocessing_test():
    """
    the ml_filter
        test ml_filter's pass_num, process_num, pass_rate, cost
        test ml_operator's cost
    """
    scan = ImageJsonScan(IMAGE_COCO_TRAIN_DATA_PATH)
    block_records = scan.get_block_records(100)
    operator = ImageObjectDetectionYolov3()
    time1 = time.time()

    def f(result: Any):
        return any(
            [key in ['person', 'dog', 'bird', 'cat', 'horse', 'sheep', 'cow'] for key in result.keys()])

    ml_predicate1 = MLPredicator(ml_operator=operator, predicate=f)
    ml_filter_1 = MLFilter(ml_predicator=ml_predicate1)

    batch_output1 = ml_filter_1.multiple_processes_batch(batch=block_records, num_process=2)
    ml_filter_1.get_operator_cost()
    ml_filter_1.print_operator_detail()
    ml_filter_1.ml_predicator.print_operator_detail()
    ml_filter_1.ml_predicator.ml_operator.print_operator_detail()
    print("the number of records after " + ml_filter_1.operator_name + " = " + str(len(batch_output1.returned_batch)))
    scan.close()
    time2 = time.time()
    print("multithreading cost = " + str(time2 - time1) + "\n")


def operator_copy_test():
    time0 = time.time()
    pos_tagger = POSTaggerSpacySM()

    def f(result: Any):
        return any([value[0] in ['VBD', 'WRB', 'PRP'] for token, value in result.items()])

    ml_predicate1 = MLPredicator(ml_operator=pos_tagger, predicate=f)
    ml_filter = MLFilter(ml_predicator=ml_predicate1)
    time1 = time.time()
    ml_filter_copy = ml_filter.get_new_object()
    print("create model time = " + str(time1 - time0) + "\tcopy model time = " + str(time.time() - time1))

    print("ml_filter.process_count = " + str(
        ml_filter.process_count) + "\tml_filter.ml_predicate.ml_operator.process_count = " + str(
        ml_filter.ml_predicator.ml_operator.process_count))
    print("ml_filter_copy.process_count = " + str(
        ml_filter_copy.process_count) + "\tml_filter_copy.ml_predicate.ml_operator.process_count = " + str(
        ml_filter_copy.ml_predicator.ml_operator.process_count))

    ml_filter.ml_predicator.ml_operator.process_count = 1000
    print("ml_filter.process_count = " + str(
        ml_filter.process_count) + "\tml_filter.ml_predicate.ml_operator.process_count = " + str(
        ml_filter.ml_predicator.ml_operator.process_count))
    print("ml_filter_copy.process_count = " + str(
        ml_filter_copy.process_count) + "\tml_filter_copy.ml_predicate.ml_operator.process_count = " + str(
        ml_filter_copy.ml_predicator.ml_operator.process_count))
    ml_filter_copy.ml_predicator.ml_operator.process_count = 2000
    print("ml_filter.process_count = " + str(
        ml_filter.process_count) + "\tml_filter.ml_predicate.ml_operator.process_count = " + str(
        ml_filter.ml_predicator.ml_operator.process_count))
    print("ml_filter_copy.process_count = " + str(
        ml_filter_copy.process_count) + "\tml_filter_copy.ml_predicate.ml_operator.process_count = " + str(
        ml_filter_copy.ml_predicator.ml_operator.process_count))
    ml_filter.operator_reset()
    print("ml_filter.process_count = " + str(
        ml_filter.process_count) + "\tml_filter.ml_predicate.ml_operator.process_count = " + str(
        ml_filter.ml_predicator.ml_operator.process_count))
    print("ml_filter_copy.process_count = " + str(
        ml_filter_copy.process_count) + "\tml_filter_copy.ml_predicate.ml_operator.process_count = " + str(
        ml_filter_copy.ml_predicator.ml_operator.process_count))
    ml_filter_copy.operator_reset()
    print("ml_filter.process_count = " + str(
        ml_filter.process_count) + "\tml_filter.ml_predicate.ml_operator.process_count = " + str(
        ml_filter.ml_predicator.ml_operator.process_count))
    print("ml_filter_copy.process_count = " + str(
        ml_filter_copy.process_count) + "\tml_filter_copy.ml_predicate.ml_operator.process_count = " + str(
        ml_filter_copy.ml_predicator.ml_operator.process_count))


if __name__ == '__main__':
    pass
#     ml_filter_test()
#     multithreading_test()
#     multiprocessing_test()
#     operator_copy_test()
