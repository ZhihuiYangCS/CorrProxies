import time
from typing import List

import rootpath

rootpath.append()
from utility.utility_base import copy_filter_batch_out_result_all
from utility.constant import TWITTER_TYPE, UCF101_TYPE
from utility.utility import batch_records_delete_ml_operator_results, sample_train_stop_condition
from records.record import Record


class WorkflowLabeledSample:
    """
    this class to get labeled samples for a whole workflow
    """

    def __init__(self, workflow):
        super().__init__()
        self.workflow = workflow

    def generate_workflow_samples(self) -> List[Record]:
        """
        generate workflow level labeled samples
        """
        time0 = time.time()
        scan, sink = self.workflow.get_scan_sink()
        ml_filters = self.workflow.get_ml_filters()
        ml_operators = self.workflow.get_ml_operators()
        last_filter_process_num = 0
        last_filter_pos_num = 0
        workflow_samples = []
        scan.operator_reset()
        batch_records = scan.get_block_records(self.workflow.batch_size)
        stop_flag = False
        time1 = time.time()
        while not stop_flag and batch_records:
            if scan.output_num % self.workflow.batch_size == 0:
                print(scan.output_num)
            block_records = batch_records.copy()
            # workflow_samples.extend(block_records)
            for ml_filter in ml_filters[:-1]:
                if self.workflow.workflow_type == UCF101_TYPE and self.workflow.opt_thread_num == 0:
                    batch_output = ml_filter.process_batch(batch=block_records, model=self.workflow.models[
                        ml_filter.ml_predicator.ml_operator.operator_name])
                elif self.workflow.workflow_type == TWITTER_TYPE:
                    batch_output = ml_filter.multiple_threads_batch(batch=block_records,
                                                                    num_process=self.workflow.opt_thread_num,
                                                                    model=self.workflow.models[
                                                                        ml_filter.ml_predicator.ml_operator.operator_name])
                else:
                    batch_output = ml_filter.multiple_processes_batch(batch=block_records,
                                                                      num_process=self.workflow.opt_thread_num)
                copy_filter_batch_out_result_all(large_batch_records=batch_records,
                                                 small_batch_records_has_results=block_records)
                block_records = batch_output.returned_batch
            last_filter_process_num += len(block_records)
            if self.workflow.workflow_type == UCF101_TYPE and self.workflow.opt_thread_num == 0:
                batch_output = ml_filters[-1].process_batch(batch=block_records, model=self.workflow.models[
                    ml_filters[-1].ml_predicator.ml_operator.operator_name])
            elif self.workflow.workflow_type == TWITTER_TYPE:
                batch_output = ml_filters[-1].multiple_threads_batch(batch=block_records,
                                                                     num_process=self.workflow.opt_thread_num,
                                                                     model=self.workflow.models[ml_filters[
                                                                         -1].ml_predicator.ml_operator.operator_name])
            else:
                batch_output = ml_filters[-1].multiple_processes_batch(batch=block_records,
                                                                       num_process=self.workflow.opt_thread_num)
            copy_filter_batch_out_result_all(large_batch_records=batch_records,
                                             small_batch_records_has_results=block_records)
            block_records = batch_output.returned_batch
            last_filter_pos_num += len(block_records)
            stop_flag = sample_train_stop_condition(workflow_type=self.workflow.workflow_type,
                                                    process_num=last_filter_process_num, pos_num=last_filter_pos_num)
            batch_records_delete_ml_operator_results(batch_records=batch_records, ml_operators=ml_operators)
            workflow_samples.extend(batch_records)
            batch_records = scan.get_block_records(self.workflow.batch_size)
            for ml_filter in ml_filters:
                print(ml_filter.operator_name + " pass rate = " + str(ml_filter.get_operator_pass_rate()))
            print("last_filter_process_num = " + str(last_filter_process_num) + "\tlast_filter_pos_num = " + str(
                last_filter_pos_num))
        # multiple threads
        time2 = time.time()
        print("get labeled sample cost = " + str(time2 - time1))
        if self.workflow.workflow_type == UCF101_TYPE and self.workflow.opt_thread_num == 0:
            preprocessor_batch_ouput = self.workflow.preprocessor.process_batch(batch=workflow_samples,
                                                                                model=self.workflow.models[
                                                                                    self.workflow.preprocessor.operator_name])
        else:
            preprocessor_batch_ouput = self.workflow.preprocessor.multiple_processes_batch(batch=workflow_samples,
                                                                                           num_process=self.workflow.opt_thread_num)
        workflow_samples = preprocessor_batch_ouput.returned_batch
        time3 = time.time()
        print("preprocess cost = " + str(time3 - time2))
        for ml_filter in ml_filters:
            print(ml_filter.operator_name + " pass rate = " + str(ml_filter.get_operator_pass_rate()))
        self.workflow.get_ml_operator_costs()
        time4 = time.time()
        print("workflow labeled sample: generate_workflow_samples cost = " + str(time4 - time0))
        return workflow_samples
