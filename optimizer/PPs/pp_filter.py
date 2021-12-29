import time
from typing import Any, List

import numpy
import rootpath

rootpath.append()
from optimizer.PPs.preprocessor.preprocessor import Preprocessor
from operators.operator_base.operator_parallel import OperatorParallel
from optimizer.PPs.accuracy_infos import AccuracyInfoDict
from records.record import Record
from operators.operator_base.operator_utility import BatchOutput


class PPFilter(OperatorParallel):
    """a pp_filter operator"""

    def __init__(self, sub_plan: List[int], estimator, accuracy_info_dict: AccuracyInfoDict,
                 preprocessor: Preprocessor, operator_name: str = ''):
        operator_name += '_'.join(map(lambda x: str(x), sub_plan))
        super().__init__(operator_name)
        self.sub_plan = sub_plan
        self.estimator = estimator
        self.accuracy_info_dict = accuracy_info_dict
        self.preprocessor = preprocessor
        self.pos_class_index = int(numpy.where(self.estimator.classes_ == 1)[0][0])

    def get_operator_cost(self):
        """return operator time cost per record, in ms/record
            :raises ValueError
        """
        if self.operator_cost is None:
            if self.process_count == 0:
                print("Getting operator cost: process_count = " + str(
                    self.process_count) + "\tYou may forget add processing statistic informations ...")
                self.operator_cost = None
            else:
                self.operator_cost = self.process_time / self.process_count

    def load_model(self) -> Any:
        return self.preprocessor.load_model()

    def _process(self, one: Record, model=None) -> Any:
        """
        pp_filter process one record. It first converts text to feature, then predict_proba by estimator
        caculate pp_filter's cost and pass_rate
        :param one:
        :return:
        """
        pass

    def filter_process(self, batch: List[Record], num_process: int = 0, model: Any = None,
                       mode: int = 0) -> BatchOutput:
        returned_batch, processed_count, passed_count, processed_time = [], 0, 0, 0
        if mode == 0:
            preprocessor_batch_output = self.preprocessor.process_one(record=batch[0], model=model)
        elif mode == 1:
            preprocessor_batch_output = self.preprocessor.process_batch(batch=batch, model=model)
        elif mode == 2:
            preprocessor_batch_output = self.preprocessor.multiple_threads_batch(batch=batch, num_process=num_process,
                                                                                 model=model)
        else:
            preprocessor_batch_output = self.preprocessor.multiple_processes_batch(batch=batch, num_process=num_process)
        start_time = time.time()
        for record in preprocessor_batch_output.returned_batch:
            processed_count += 1
            feature = self.preprocessor.get_result(one=record)
            y_pred_pos_proba = self.estimator.predict_proba(feature)[:, self.pos_class_index]

            if y_pred_pos_proba >= self.accuracy_info_dict.threshold:
                passed_count += 1
                returned_batch.append(record)
        end_time = time.time()
        if self.operator_cost is None:
            # processed_time = preprocessor_batch_output.processed_time + (end_time - start_time)
            processed_time = end_time - start_time
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

    def multiple_processes_batch(self, batch: List[Record], num_process: int) -> BatchOutput:
        return self.filter_process(batch=batch, num_process=num_process, mode=3)

    def multiple_threads_batch(self, batch: List[Record], num_process: int, model=None) -> BatchOutput:
        return self.filter_process(batch=batch, num_process=num_process, model=model, mode=2)

    def get_new_object(self, previous_operators=None, operator_name: str = ""):
        """
        return a new object
        :param previous_operators: List[Operator]
        :param operator_name: str, default value is ""
        """
        new_filter = PPFilter(sub_plan=self.sub_plan, estimator=self.estimator,
                              accuracy_info_dict=self.accuracy_info_dict, preprocessor=self.preprocessor,
                              operator_name=operator_name)
        if self.operator_cost is not None:
            new_filter.operator_cost = self.operator_cost
        return new_filter
