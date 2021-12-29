import time
from typing import Any, List

import rootpath


rootpath.append()
from utility.utility_base import copy_filter_batch_out_result_all
from operators.ml_operators.image_video_operators.image_object_detection_yolov3 import ImageObjectDetectionYolov3
from operators.scan.image_json_scan import ImageJsonScan
from operators.operator_base.operator_utility import BatchOutput
from operators.scan.text_json_scan import TextJsonScan
from paths import TEXT_TWITTER_TRAIN_DATA_PATH, IMAGE_COCO_TRAIN_DATA_PATH
from records.record import Record
from operators.ml_operators.text_operators.spacy.spacy_operator import POSTaggerSpacySM
from operators.operator_base.operator_parallel import OperatorParallel


class MLPredicator(OperatorParallel):
    """
    Labels the given Record to be 1 (True) or 0 (False), based on the provided predicate
    """

    def __init__(self, ml_operator: OperatorParallel, operator_name: str = '', predicate=lambda r: r, *args,
                 **kwargs):
        super().__init__(f'{ml_operator.operator_name}_{predicate.__name__}_{operator_name}')
        self.predicate = predicate
        self.ml_operator = ml_operator
        self.args = args
        self.kwargs = kwargs

    def get_operator_cost(self):
        """return operator time cost per record, in ms/record
            :raises ValueError
        """
        self.ml_operator.get_operator_cost()
        if self.operator_cost is None:
            if self.process_count == 0:
                print("Getting operator cost: process_count = " + str(
                    self.process_count) + "\tYou may forget add processing statistic informations ...")
                self.operator_cost = None
            else:
                self.operator_cost = self.process_time / self.process_count

    def load_model(self) -> Any:
        return self.ml_operator.load_model()

    def _process(self, one: Record, model=None) -> Any:
        pass

    def predict_process(self, batch: List[Record], num_process: int = 0, model: Any = None,
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
        processed_batch, unprocessed_batch_index, unprocessed_batch = {}, [], []
        for index, record in enumerate(batch):
            if self.get_result(record) is None:
                unprocessed_batch_index.append(index)
                unprocessed_batch.append(record)
            else:
                processed_batch[index] = record
        if unprocessed_batch:
            if mode == 0:
                ml_operator_batch_output = self.ml_operator.process_one(record=unprocessed_batch[0], model=model)
            elif mode == 1:
                ml_operator_batch_output = self.ml_operator.process_batch(batch=unprocessed_batch, model=model)
            elif mode == 2:
                ml_operator_batch_output = self.ml_operator.multiple_threads_batch(batch=unprocessed_batch,
                                                                                   num_process=num_process, model=model)
            else:
                ml_operator_batch_output = self.ml_operator.multiple_processes_batch(batch=unprocessed_batch,
                                                                                     num_process=num_process)
            start_time = time.time()
            for record in ml_operator_batch_output.returned_batch:
                operator_result = self.ml_operator.get_result(one=record)
                predict_result = int(bool(self.predicate(operator_result, *self.args, **self.kwargs)))
                self.set_result(one=record, result=predict_result)
            end_time = time.time()
            if self.operator_cost is None:
                processed_time = ml_operator_batch_output.processed_time + (end_time - start_time)
            else:
                processed_time = self.operator_cost * len(unprocessed_batch)
            processed_count = len(unprocessed_batch)
            passed_count = len(unprocessed_batch)
            for i in range(len(unprocessed_batch_index)):
                processed_batch[unprocessed_batch_index[i]] = ml_operator_batch_output.returned_batch[i]
        else:
            processed_count = 0
            passed_count = 0
            processed_time = 0

        returned_batch = [processed_batch[key] for key in sorted(processed_batch.keys())]
        copy_filter_batch_out_result_all(large_batch_records=batch,
                                         small_batch_records_has_results=returned_batch)
        returned_batch_output = BatchOutput(returned_batch=returned_batch, processed_count=processed_count,
                                            passed_count=passed_count, processed_time=processed_time)
        self.set_processed_statistics(batch_output=returned_batch_output)
        return returned_batch_output

    def process_one(self, record: Record, model=None) -> BatchOutput:
        return self.predict_process(batch=[record], model=model, mode=0)

    def process_batch(self, batch: List[Record], model=None) -> BatchOutput:
        return self.predict_process(batch=batch, model=model, mode=1)

    def multiple_processes_batch(self, batch: List[Record], num_process: int) -> BatchOutput:
        return self.predict_process(batch=batch, num_process=num_process, mode=3)

    def multiple_threads_batch(self, batch: List[Record], num_process: int, model=None) -> BatchOutput:
        return self.predict_process(batch=batch, num_process=num_process, model=model, mode=2)

    def operator_reset(self):
        self.process_time = 0
        self.pass_count = 0
        self.process_count = 0
        self.ml_operator.operator_reset()

    def get_new_object(self, previous_operators=None, operator_name: str = ""):
        """
        return a new object
        :param previous_operators: List[Operator]
        :param operator_name: str, default value is ""
        """
        if previous_operators is not None and previous_operators and previous_operators[
            -1].ml_predicator.ml_operator.operator_name == self.ml_operator.operator_name:
            ml_operator = previous_operators[-1].ml_predicator.ml_operator
        else:
            ml_operator = self.ml_operator.get_new_object()
        new_ml_predict = MLPredicator(ml_operator=ml_operator, predicate=self.predicate, operator_name=operator_name)
        if self.operator_cost is not None:
            new_ml_predict.operator_cost = self.operator_cost
        return new_ml_predict


"""
test units
"""


def ml_predicator_test():
    """
    test ml_predicator:
        test if ml_predicator's name is correct
        test ml_predicator cost and ml_operator cost
        test ml_predicator result
    """
    scan = TextJsonScan(TEXT_TWITTER_TRAIN_DATA_PATH)
    block_records = scan.get_block_records(100)
    pos_tagger = POSTaggerSpacySM()
    time1 = time.time()

    def f(result: Any):
        return any([value[0] in ['VBD', 'WRB', 'PRP'] for token, value in result.items()])

    ml_predicate1 = MLPredicator(ml_operator=pos_tagger, predicate=f)
    batch_output = ml_predicate1.process_batch(batch=block_records, model=ml_predicate1.load_model())
    ml_predicate1.get_operator_cost()
    ml_predicate1.print_operator_detail()
    ml_predicate1.ml_operator.print_operator_detail()
    print(batch_output)
    print(batch_output.returned_batch[0])
    time2 = time.time()

    def g(result: Any):
        return any([value[0] in ['VBD', 'WRB', 'PRP'] for token, value in result.items()])

    ml_predicate2 = MLPredicator(ml_operator=pos_tagger, predicate=g)
    batch_output2 = ml_predicate2.process_batch(batch=block_records, model=ml_predicate2.load_model())
    ml_predicate2.get_operator_cost()
    ml_predicate2.print_operator_detail()
    ml_predicate2.ml_operator.print_operator_detail()
    print(batch_output2)
    print(batch_output2.returned_batch[0])
    time3 = time.time()

    scan.close()

    print("first predict time = " + str(time2 - time1) + "\t second predict time = " + str(time3 - time2) + "\n")


def multithreading_test():
    scan = TextJsonScan(TEXT_TWITTER_TRAIN_DATA_PATH)
    block_records = scan.get_block_records(100)
    pos_tagger = POSTaggerSpacySM()
    time1 = time.time()

    def f(result: Any):
        return any([value[0] in ['VBD', 'WRB', 'PRP'] for token, value in result.items()])

    ml_predicate1 = MLPredicator(ml_operator=pos_tagger, predicate=f)
    batch_output = ml_predicate1.multiple_threads_batch(batch=block_records, num_process=10,
                                                        model=ml_predicate1.load_model())
    ml_predicate1.get_operator_cost()
    ml_predicate1.print_operator_detail()
    ml_predicate1.ml_operator.print_operator_detail()
    print(batch_output)
    print(batch_output.returned_batch[0])
    time2 = time.time()
    print("multithreading cost = " + str(time2 - time1) + "\n")


def multiprocessing_test():
    scan = ImageJsonScan(IMAGE_COCO_TRAIN_DATA_PATH)
    block_records = scan.get_block_records(100)
    operator = ImageObjectDetectionYolov3()
    time1 = time.time()

    def f(result: Any):
        return any(
            [key in ['person', 'dog', 'bird', 'cat', 'horse', 'sheep', 'cow'] for key in result.keys()])

    ml_predicate1 = MLPredicator(ml_operator=operator, predicate=f)
    batch_output = ml_predicate1.multiple_processes_batch(batch=block_records, num_process=2)
    ml_predicate1.get_operator_cost()
    ml_predicate1.print_operator_detail()
    ml_predicate1.ml_operator.print_operator_detail()
    print(batch_output)
    print(batch_output.returned_batch[0])
    time2 = time.time()
    print("multiprocessing cost = " + str(time2 - time1) + "\n")


if __name__ == '__main__':
    pass
#     ml_predicator_test()
#     multithreading_test()
#     multiprocessing_test()
