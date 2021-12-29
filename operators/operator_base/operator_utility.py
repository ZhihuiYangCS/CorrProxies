from abc import ABC
from typing import List
import rootpath

rootpath.append()
from records.record import Record


class BatchOutput(ABC):
    def __init__(self, returned_batch: List[Record] = [], processed_count: int = 0, passed_count: int = 0,
                 processed_time: float = 0):
        self.returned_batch = returned_batch
        self.processed_count = processed_count
        self.passed_count = passed_count
        self.processed_time = processed_time

    def __str__(self):
        return "one batch output: processed_count = " + str(
            self.processed_count) + "\tpassed_count = " + str(self.passed_count) + "\tprocessed_time = " + str(
            self.processed_time)

    __repr__ = __str__


class OneProcessOutput(BatchOutput):
    def __init__(self, process_index: int, returned_batch: List[Record] = [], processed_count: int = 0,
                 passed_count: int = 0, processed_time: float = 0):
        super(OneProcessOutput, self).__init__(returned_batch=returned_batch, processed_count=processed_count,
                                               passed_count=passed_count, processed_time=processed_time)
        self.process_index = process_index

    def __str__(self):
        return "process_index = " + str(self.process_index) + "\t processed_count = " + str(
            self.processed_count) + "\t passed_count = " + str(self.passed_count) + "\t processed_time = " + str(
            self.processed_time)

    __repr__ = __str__


class MultipleProcessesOutput:
    def __init__(self):
        self.multiple_output_list: List[OneProcessOutput] = []

    def add(self, one_process_output: OneProcessOutput):
        self.multiple_output_list.append(one_process_output)

    def get_all_processes_output_batch(self) -> List[Record]:
        returned_batch = []
        for i in range(len(self.multiple_output_list)):
            for one_process_output in self.multiple_output_list:
                if i == one_process_output.process_index:
                    returned_batch += one_process_output.returned_batch
        return returned_batch

    def get_batch_output(self) -> BatchOutput:
        returned_batch = self.get_all_processes_output_batch()
        processed_count, passed_count, processed_time = 0, 0, 0
        for one_process_output in self.multiple_output_list:
            processed_count += one_process_output.processed_count
            passed_count += one_process_output.passed_count
            if one_process_output.processed_time > processed_time:
                processed_time = one_process_output.processed_time
        return BatchOutput(returned_batch=returned_batch, processed_count=processed_count, passed_count=passed_count,
                           processed_time=processed_time)

    def get_processed_count(self) -> int:
        processed_count = 0
        for one_process_output in self.multiple_output_list:
            processed_count += one_process_output.processed_count
        return processed_count

    def get_passed_count(self) -> int:
        passed_count = 0
        for one_process_output in self.multiple_output_list:
            passed_count += one_process_output.passed_count
        return passed_count

    def get_max_processed_time(self) -> float:
        max_time = 0
        for one_process_output in self.multiple_output_list:
            if one_process_output.processed_time > max_time:
                max_time = one_process_output.processed_time
        return max_time

    def __str__(self):
        info = '\n '.join(map(str, self.multiple_output_list))
        return f"multiple processes output list:\n[{info}]"

    __repr__ = __str__
