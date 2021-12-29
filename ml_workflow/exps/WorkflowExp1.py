import sys
from typing import List
import rootpath

rootpath.append()
from operators.scan.video_json_scan import VideoJsonScan
from paths import TEXT_TWITTER_ALL_PATH, IMAGE_COCO_ALL_PATH, VIDEO_UCF101_ALL_PATH
from optimizer.compare_methods.microsoft_pp_dynamic_programming import MicrosoftPPDP
from optimizer.reorder.reorder.ExhaustedSearch import ExhaustedSearch
from optimizer.reorder.reorder.tree.st_tree.fine_grained_accuracy_allocation import FineGrainedAA
from optimizer.PPs.preprocessor.preprocessor import Preprocessor
from ml_workflow.workflow_base import WorkflowBase
from operators.ml_filter import MLFilter
from operators.operator_base.operator_parallel import OperatorParallel
from operators.scan.image_json_scan import ImageJsonScan
from operators.scan.text_json_scan import TextJsonScan
from optimizer.PPs.pp_filter import PPFilter
from optimizer.plan_generator.plan_generator import PlanGenerator
from optimizer.compare_methods.reorder_baseline import ReorderBaseline
from utility.constant import TARGET_ACCURACY_DEFAULT, SCHEME_DEFAULT, OPT_THREAD_NUM_DEFAULT, EXE_THREAD_NUM_DEFAULT, \
    TWITTER_NUM, TWITTER_START_EXEC_NUM, COCO_START_EXEC_NUM, COCO_NUM, UCF101_NUM, UCF101_START_EXEC_NUM, SCHEME_RAW, \
    SCHEME_NS, TWITTER_DATA_SIZES_SCALABILITY, TWITTER_TYPE, SCHEME_CORE, SCHEME_COREa, SCHEME_PP, SCHEME_COREh, \
    SCHEME_REORDER, TWITTER_EXE_BATCH_SIZE, COCO_EXE_BATCH_SIZE, UCF101_EXE_BATCH_SIZE


class WorkflowExp1(WorkflowBase):
    """
    This experiment run the whole dataset to test latency
    allocation_method: exhausted_cost_comp --> 0; exhausted_cost_func --> 1;
                       hill_climbing_cost_comp --> 2; hill_climbing_cost_func --> 3
    allocation_mode: 0 ---- resample retrain; 1 ---- reuse sample, retrain; 2 ---- reuse sample, reuse PPs

    reorder_accuracy_allocator_method: AccuracyAllocatorExhaustedSearch: --> 0; AccuracyAllocatorHillClimbing: --> 1
    target_accuracy: float
    """

    def __init__(self, operators: List[OperatorParallel], preprocessor: Preprocessor, models: dict,
                 workflow_name: str = "", target_accuracy: float = TARGET_ACCURACY_DEFAULT,
                 scheme: int = SCHEME_DEFAULT, opt_thread_num: int = OPT_THREAD_NUM_DEFAULT,
                 exe_thread_num: int = EXE_THREAD_NUM_DEFAULT):
        """
        exe_thread_num: the number of threads used for execution
        twitter_data_size_scalability: used to test scalability experiments
        data_size: the whole data size
        start_exec_index: the start index to test execution cost
        """
        super().__init__(operators=operators, preprocessor=preprocessor, models=models, workflow_name=workflow_name,
                         target_accuracy=target_accuracy, scheme=scheme, opt_thread_num=opt_thread_num)
        self.exe_thread_num = exe_thread_num
        self.twitter_data_size_scalability = TWITTER_DATA_SIZES_SCALABILITY
        if isinstance(self.workflow[0], TextJsonScan):
            self.data_size = TWITTER_NUM
            self.exe_batch_size = TWITTER_EXE_BATCH_SIZE
            self.start_exec_index = TWITTER_START_EXEC_NUM
        elif isinstance(self.workflow[0], ImageJsonScan):
            self.data_size = COCO_NUM
            self.exe_batch_size = COCO_EXE_BATCH_SIZE
            self.start_exec_index = COCO_START_EXEC_NUM
        else:
            self.data_size = UCF101_NUM
            self.exe_batch_size = UCF101_EXE_BATCH_SIZE
            self.start_exec_index = UCF101_START_EXEC_NUM

    def reset_scan(self):
        if isinstance(self.workflow[0], TextJsonScan):
            scan = TextJsonScan(TEXT_TWITTER_ALL_PATH)
        elif isinstance(self.workflow[0], ImageJsonScan):
            scan = ImageJsonScan(IMAGE_COCO_ALL_PATH)
        else:
            scan = VideoJsonScan(VIDEO_UCF101_ALL_PATH)
        return scan

    def set_operator_cost(self):
        self.workflow[0].close()
        self.workflow[0] = self.reset_scan()
        scan, sink = self.get_scan_sink()
        batch_records = scan.get_block_records(self.batch_size)
        batch_output = self.preprocessor.multiple_processes_batch(batch=batch_records, num_process=1)
        batch_records = batch_output.returned_batch
        self.preprocessor.get_operator_cost()
        print(self.preprocessor.operator_name + "\t operator_cost = " + str(self.preprocessor.operator_cost))
        for ml_filter in self.get_ml_filters():
            if self.workflow_type == TWITTER_TYPE:
                batch_output = ml_filter.multiple_threads_batch(batch=batch_records, num_process=1, model=self.models[
                    ml_filter.ml_predicator.ml_operator.operator_name])
            else:
                batch_output = ml_filter.multiple_processes_batch(batch=batch_records, num_process=1)

            batch_records = batch_output.returned_batch
            ml_filter.get_operator_cost()
        self.print_ml_operator_costs()

        self.workflow[0].close()
        self.workflow[0] = self.reset_scan()
        for ml_filter in self.get_ml_filters():
            ml_filter.operator_reset()

    def optimization(self):
        pp_filters, ml_filters, time_log = None, None, ""
        if self.scheme == SCHEME_RAW:
            self.print_ml_operator_costs()
            pp_filters, ml_filters = [], self.get_ml_filters()
            self.print_ml_operator_costs()
            print("finish optimizing RAW ...")
            print()
        elif self.scheme == SCHEME_NS:
            self.print_ml_operator_costs()
            plan_generator = PlanGenerator(workflow=self)
            pp_filters, time_log = plan_generator.one_plan_accuracy_allocation(
                plan=plan_generator.generate_one_single_plan())
            ml_filters = self.get_ml_filters()
            self.print_ml_operator_costs()
            print("finish optimizing NS ...")
            print()
        elif self.scheme == SCHEME_PP:
            self.print_ml_operator_costs()
            microsoft_pp = MicrosoftPPDP(workflow=self)
            pp_filters, time_log = microsoft_pp.multiple_pps_accuracy_allocation()
            ml_filters = self.get_ml_filters()
            self.print_ml_operator_costs()
            print("finish optimizing PP ...")
            print()
        elif self.scheme == SCHEME_CORE:
            self.print_ml_operator_costs()
            schema_method = FineGrainedAA(workflow=self)
            pp_filters, ml_filters, time_log = schema_method.reorder_optimization()
            self.print_ml_operator_costs()
            print("finish optimizing CORE ...")
            print()
        elif self.scheme == SCHEME_COREa:
            self.print_ml_operator_costs()
            plan_generator = PlanGenerator(workflow=self)
            pp_filters, time_log = plan_generator.one_plan_accuracy_allocation(
                plan=plan_generator.generate_each_pp_plan())
            ml_filters = self.get_ml_filters()
            self.print_ml_operator_costs()
            print("finish optimizing COREa ...")
            print()
        elif self.scheme == SCHEME_COREh:
            self.print_ml_operator_costs()
            schema_method = ExhaustedSearch(workflow=self)
            pp_filters, ml_filters, time_log = schema_method.reorder_optimization()
            self.print_ml_operator_costs()
            print("finish optimizing COREh ...")
            print()
        elif self.scheme == SCHEME_REORDER:
            self.print_ml_operator_costs()
            schema_method = ReorderBaseline(workflow=self)
            pp_filters, ml_filters, time_log = schema_method.reorder_optimization()
            self.print_ml_operator_costs()
            print("finish optimizing REORDER ...")
            print()
        else:
            print("Workflow does not has the schema " + str(self.scheme))
            sys.exit(0)
        scan_opt_output_num = self.workflow[0].output_num
        print("optimization details ...")
        print("time_log = " + str(time_log) + "\t scan_opt_output_num = " + str(scan_opt_output_num))
        return pp_filters, ml_filters, time_log, scan_opt_output_num

    def execution(self, pp_filters: List[PPFilter], ml_filters: List[MLFilter]):
        batch_records, execution_latency, estimated_execution_costs_scalability, estimated_execution_cost, start_estimated_execution_cost = [], 0, [], 0, 0
        while self.workflow[0].output_num % self.exe_batch_size != 0:
            batch_records.append(self.workflow[0].get_one_record())
        batch_records += self.workflow[0].get_block_records(self.exe_batch_size)
        output_records = []
        while batch_records and self.workflow[0].output_num <= self.data_size:
            input_batch_size = len(batch_records)
            batch_estimated_time = 0
            for pp_filter in pp_filters:
                batch_estimated_time += len(batch_records) * pp_filter.accuracy_info_dict.infer_cost
                pp_batch_output = pp_filter.multiple_processes_batch(batch=batch_records,
                                                                     num_process=self.exe_thread_num)
                batch_records = pp_batch_output.returned_batch
            for ml_filter in ml_filters:
                if self.workflow_type == TWITTER_TYPE:
                    ml_batch_output = ml_filter.multiple_threads_batch(batch=batch_records,
                                                                       num_process=self.exe_thread_num,
                                                                       model=self.models[
                                                                           ml_filter.ml_predicator.ml_operator.operator_name])
                else:
                    ml_batch_output = ml_filter.multiple_processes_batch(batch=batch_records,
                                                                         num_process=self.exe_thread_num)
                batch_estimated_time += ml_batch_output.processed_time
                batch_records = ml_batch_output.returned_batch
            execution_latency += batch_estimated_time
            print("scan output num = " + str(self.workflow[0].output_num) + "\tprocess num = " + str(
                input_batch_size) + "\tpass_num = " + str(len(batch_records)) + "\tpass_rate = " + str(
                len(batch_records) / input_batch_size) + "\t estimated_time = " + str(
                batch_estimated_time / input_batch_size))
            if self.workflow_type == TWITTER_TYPE and self.workflow[0].output_num in self.twitter_data_size_scalability:
                estimated_execution_costs_scalability.append(execution_latency)
            if self.workflow[0].output_num == self.start_exec_index:
                start_estimated_execution_cost = execution_latency
                output_records.extend(batch_records.copy())
            batch_records = self.workflow[0].get_block_records(self.exe_batch_size)
        estimated_execution_cost = (execution_latency - start_estimated_execution_cost) / (
                self.data_size - self.start_exec_index)
        return execution_latency, estimated_execution_costs_scalability, estimated_execution_cost, len(output_records)

    def run(self):
        self.set_operator_cost()
        pp_filters, ml_filters, time_log, scan_opt_output_num = self.optimization()
        execution_latency, estimated_execution_costs_scalability, estimated_execution_cost, output_num = self.execution(
            pp_filters=pp_filters, ml_filters=ml_filters)
        print("optimization cost = " + time_log)
        print("optimization used output num = " + str(scan_opt_output_num))
        # the following estimated_exeuction_latency is the estimated latency during the whole optimization and execution
        print("estimated_execution_latency = " + str(execution_latency))
        # the following estimated_execution_costs_scalability does not include the optimization cost
        print("execution estimated execution costs = " + str(estimated_execution_costs_scalability))
        # the following estimated_execution_cost is an estimated execution cost for the last several blocks
        print("estimated_execution_cost = " + str(estimated_execution_cost))
        # the following output_num is the last several blocks output records number, it is used to compute accuracy
        print("execution output_num = " + str(output_num))
