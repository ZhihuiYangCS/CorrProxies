import time
from abc import abstractmethod
from typing import Any, List
import rootpath

rootpath.append()
from operators.operator_base.operator_utility import BatchOutput
from operators.operator_base.operator_base import OperatorStaticBase
from records.record import Record


class OperatorDynamicAction(OperatorStaticBase):
    """
    this class implements the process method for an operator, including process_one and process_batch
    for process_one and process_batch, they can be called outside of this class;
    and for all for them have process_one_inner and process_batch_inner methods;
    these inner methods are used inside this class.
    """
    def __init__(self, operator_name: str = ""):
        super(OperatorDynamicAction, self).__init__(operator_name=operator_name)

    @abstractmethod
    def load_model(self) -> Any:
        """
        loading an ML model, which is used for process one record
        """
        pass

    @abstractmethod
    def _process(self, one: Record, model=None) -> Any:
        """base method to get result of one record"""
        pass

    def process_one_inner(self, record: Record, model=None) -> BatchOutput:
        """
        process one record, return processed record, or None, and processing statistics,
        such as processed_count, passed_count and processed_time
        """
        # if the record already has the ml operator inference result, we reuse the result withouts compute it again.
        if self.get_result(record) is None:
            start_time = time.time()
            # if the record does not has the ml operator inference result, we need to compute it
            self.set_result(record, self._process(one=record, model=model))
            end_time = time.time()
            if self.operator_cost is None:
                processed_time = end_time - start_time
            else:
                processed_time = self.operator_cost
            processed_count = 1
            if record is not None:
                passed_count = 1
            else:
                passed_count = 0
        else:
            processed_time = 0
            processed_count = 0
            passed_count = 0
        returned_batch = []
        if record is not None:
            returned_batch.append(record)
        return BatchOutput(returned_batch=returned_batch, processed_count=processed_count, passed_count=passed_count,
                           processed_time=processed_time)

    def process_one(self, record: Record, model=None) -> BatchOutput:
        batch_output = self.process_one_inner(record=record, model=model)
        self.set_processed_statistics(batch_output=batch_output)
        return batch_output

    def process_batch_inner(self, batch: List[Record], model=None) -> BatchOutput:
        """
        process a batch of records, return a list of records and processing statistics,
            such as processed_count, passed_count and processed_time
        """
        returned_batch, processed_count, passed_count, processed_time = [], 0, 0, 0
        for record in batch:
            one_output = self.process_one_inner(record=record, model=model)
            returned_batch += one_output.returned_batch
            processed_count += one_output.processed_count
            passed_count += one_output.passed_count
            processed_time += one_output.processed_time
        return BatchOutput(returned_batch=returned_batch, processed_count=processed_count, passed_count=passed_count,
                           processed_time=processed_time)

    def process_batch(self, batch: List[Record], model=None) -> BatchOutput:
        """
        process a batch of records, return a list of records and processing statistics,
            such as processed_count, passed_count and processed_time
        """
        batch_output = self.process_batch_inner(batch=batch, model=model)
        self.set_processed_statistics(batch_output=batch_output)
        return batch_output
