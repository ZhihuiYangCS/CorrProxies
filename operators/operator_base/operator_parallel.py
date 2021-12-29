import gc
import math
from abc import ABC
from typing import List
import concurrent.futures
import rootpath

rootpath.append()
from operators.operator_base.operator_utility import BatchOutput, OneProcessOutput, MultipleProcessesOutput
from operators.operator_base.operator_action import OperatorDynamicAction
from records.record import Record
from utility.utility_base import copy_filter_batch_out_result_all


class OperatorParallel(OperatorDynamicAction, ABC):
    def __init__(self, operator_name: str = ""):
        super(OperatorParallel, self).__init__(operator_name=operator_name)

    def one_process_batch(self, process_index: int, batch: List[Record]):
        model = self.load_model()
        batch_out = self.process_batch_inner(batch=batch, model=model)
        # model.close()
        return OneProcessOutput(process_index=process_index, returned_batch=batch_out.returned_batch,
                                processed_count=batch_out.processed_count, passed_count=batch_out.passed_count,
                                processed_time=batch_out.processed_time)

    def multiple_processes_batch_inner(self, batch: List[Record], num_process: int) -> BatchOutput:
        interval_size = math.ceil(len(batch) / num_process)
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = []
            for i in range(num_process):
                if (i + 1) * interval_size > len(batch):
                    futures.append(executor.submit(self.one_process_batch, process_index=i,
                                                   batch=batch[i * interval_size:len(batch)]))
                else:
                    futures.append(executor.submit(self.one_process_batch, process_index=i,
                                                   batch=batch[i * interval_size:(i + 1) * interval_size]))
            # release_list_records(batch)
            multiple_process_output = MultipleProcessesOutput()
            for future in concurrent.futures.as_completed(futures):
                one_process_output = future.result()
                multiple_process_output.add(one_process_output=one_process_output)
                del future
            batch_output = multiple_process_output.get_batch_output()
        self.set_processed_statistics(batch_output=batch_output)
        copy_filter_batch_out_result_all(large_batch_records=batch,
                                         small_batch_records_has_results=batch_output.returned_batch)
        gc.collect()
        return batch_output

    def multiple_processes_batch(self, batch: List[Record], num_process: int) -> BatchOutput:
        return self.multiple_processes_batch_inner(batch=batch, num_process=num_process)

    def one_thread_batch(self, process_index: int, batch: List[Record], model=None):
        # model = self.load_model()
        batch_out = self.process_batch_inner(batch=batch, model=model)
        # model.close()
        return OneProcessOutput(process_index=process_index, returned_batch=batch_out.returned_batch,
                                processed_count=batch_out.processed_count, passed_count=batch_out.passed_count,
                                processed_time=batch_out.processed_time)

    def multiple_threads_batch_inner(self, batch: List[Record], num_process: int, model=None) -> BatchOutput:
        interval_size = math.ceil(len(batch) / num_process)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            for i in range(num_process):
                if (i + 1) * interval_size > len(batch):
                    futures.append(executor.submit(self.one_thread_batch, process_index=i, model=model,
                                                   batch=batch[i * interval_size:len(batch)]))
                else:
                    futures.append(executor.submit(self.one_thread_batch, process_index=i, model=model,
                                                   batch=batch[i * interval_size:(i + 1) * interval_size]))
            # release_list_records(batch)
            multiple_process_output = MultipleProcessesOutput()
            for future in concurrent.futures.as_completed(futures):
                one_process_output = future.result()
                multiple_process_output.add(one_process_output=one_process_output)
                del future
            batch_output = multiple_process_output.get_batch_output()
        self.set_processed_statistics(batch_output=batch_output)
        copy_filter_batch_out_result_all(large_batch_records=batch,
                                         small_batch_records_has_results=batch_output.returned_batch)
        # gc.collect()
        return batch_output

    def multiple_threads_batch(self, batch: List[Record], num_process: int, model=None) -> BatchOutput:
        return self.multiple_threads_batch_inner(batch=batch, num_process=num_process, model=model)
