from typing import List
import rootpath

rootpath.append()
from utility.utility_base import copy_filter_batch_out_result_all
from operators.operator_base.operator_parallel import OperatorParallel
from ml_workflow.workflow_base import WorkflowBase
from operators.ml_filter import MLFilter
from operators.scan.image_json_scan import ImageJsonScan
from operators.scan.text_json_scan import TextJsonScan
from operators.scan.video_json_scan import VideoJsonScan
from optimizer.PPs.pp_filter import PPFilter
from optimizer.PPs.preprocessor.preprocessor import Preprocessor
from optimizer.compare_methods.microsoft_pp_dynamic_programming import MicrosoftPPDP
from optimizer.compare_methods.reorder_baseline import ReorderBaseline
from optimizer.plan_generator.plan_generator import PlanGenerator
from optimizer.reorder.reorder.ExhaustedSearch import ExhaustedSearch
from optimizer.reorder.reorder.tree.st_tree.fine_grained_accuracy_allocation import FineGrainedAA
from paths import TEXT_TWITTER_TRAIN_DATA_PATH, IMAGE_COCO_TRAIN_DATA_PATH, VIDEO_UCF101_TRAIN_DATA_PATH, \
    TEXT_TWITTER_VALI_DATA_PATH, IMAGE_COCO_VALI_DATA_PATH, VIDEO_UCF101_VALI_DATA_PATH
from utility.constant import TARGET_ACCURACY_DEFAULT, SCHEME_DEFAULT, OPT_THREAD_NUM_DEFAULT, EXE_THREAD_NUM_DEFAULT, \
    SCHEME_RAW, SCHEME_NS, SCHEME_PP, SCHEME_CORE, SCHEME_COREa, SCHEME_COREh, SCHEME_REORDER, TWITTER_TYPE, \
    COCO_TYPE, COCO_EXE_BATCH_SIZE, UCF101_EXE_BATCH_SIZE, TWITTER_EXE_BATCH_SIZE
from utility.utility import batch_records_delete_ml_operator_results


class WorkflowExp3(WorkflowBase):
    """
    This experiment is used to test different schemes optimization overhead, throughput and accuracy
    """

    def __init__(self, operators: List[OperatorParallel], preprocessor: Preprocessor, models: dict,
                 workflow_name: str = "", target_accuracy: float = TARGET_ACCURACY_DEFAULT,
                 scheme: int = SCHEME_DEFAULT, opt_thread_num: int = OPT_THREAD_NUM_DEFAULT,
                 exe_thread_num: int = EXE_THREAD_NUM_DEFAULT):
        """
        exe_thread_num: the number of threads used for execution
        exe_batch_size: the number of records in a batch during execution phase.
                        a bigger batch_size than optimization
        """
        super().__init__(operators=operators, preprocessor=preprocessor, models=models, workflow_name=workflow_name,
                         target_accuracy=target_accuracy, scheme=scheme, opt_thread_num=opt_thread_num)
        self.exe_thread_num = exe_thread_num
        if self.workflow_type == TWITTER_TYPE:
            self.exe_batch_size = TWITTER_EXE_BATCH_SIZE
        elif self.workflow_type == COCO_TYPE:
            self.exe_batch_size = COCO_EXE_BATCH_SIZE
        else:
            self.exe_batch_size = UCF101_EXE_BATCH_SIZE

    def reset_scan(self, train_flag: bool):
        scan = None
        if train_flag:
            if isinstance(self.workflow[0], TextJsonScan):
                scan = TextJsonScan(TEXT_TWITTER_TRAIN_DATA_PATH)
            if isinstance(self.workflow[0], ImageJsonScan):
                scan = ImageJsonScan(IMAGE_COCO_TRAIN_DATA_PATH)
            if isinstance(self.workflow[0], VideoJsonScan):
                scan = VideoJsonScan(VIDEO_UCF101_TRAIN_DATA_PATH)
        else:
            if isinstance(self.workflow[0], TextJsonScan):
                scan = TextJsonScan(TEXT_TWITTER_VALI_DATA_PATH)
            if isinstance(self.workflow[0], ImageJsonScan):
                scan = ImageJsonScan(IMAGE_COCO_VALI_DATA_PATH)
            if isinstance(self.workflow[0], VideoJsonScan):
                scan = VideoJsonScan(VIDEO_UCF101_VALI_DATA_PATH)
        return scan

    def set_operators_cost(self):
        """
        We use different number of multiple threads in optimization phase and execution phase.
        We need to set our operators costs using our execution threads number before optimization phase,
            because we need to use operator_costs to allocate accuracy.
        pay attention to ML_Operators cost as they are class level.
        """
        self.workflow[0].close()
        self.workflow[0] = self.reset_scan(train_flag=True)
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

    def optimization(self):
        schemes_list, pp_filters_list, ml_filters_list, time_logs_list, scan_opt_output_num_list = [], [], [], [], []
        # SCHEME_RAW
        workflow_raw = self.get_new_workflow()
        workflow_raw.workflow[0].close()
        workflow_raw.workflow[0] = self.reset_scan(train_flag=True)
        workflow_raw.print_ml_operator_costs()
        schemes_list.append(SCHEME_RAW)
        pp_filters_list.append([])
        ml_filters_list.append(workflow_raw.get_ml_filters())
        time_logs_list.append("")
        scan_opt_output_num_list.append(0)
        scan_raw, sink_raw = workflow_raw.get_scan_sink()
        scan_raw.close()
        sink_raw.close()
        workflow_raw.print_ml_operator_costs()
        print("finish optimizing RAW ...")
        print()

        if self.scheme == SCHEME_NS:  # SCHEME_NS
            workflow_NS = self.get_new_workflow()
            workflow_NS.workflow[0].close()
            workflow_NS.workflow[0] = self.reset_scan(train_flag=True)
            workflow_NS.print_ml_operator_costs()
            schemes_list.append(SCHEME_NS)
            plan_generator_ns = PlanGenerator(workflow=workflow_NS)
            pp_filters_ns, time_log_ns = plan_generator_ns.one_plan_accuracy_allocation(
                plan=plan_generator_ns.generate_one_single_plan())
            pp_filters_list.append(pp_filters_ns)
            ml_filters_list.append(workflow_NS.get_ml_filters())
            time_logs_list.append(time_log_ns)
            scan_opt_output_num_list.append(workflow_NS.workflow[0].output_num)
            scan_ns, sink_ns = workflow_NS.get_scan_sink()
            scan_ns.close()
            sink_ns.close()
            workflow_NS.print_ml_operator_costs()
            print("finish optimizing NS ...")
            print()
        elif self.scheme == SCHEME_PP:  # SCHEME_PP
            workflow_pp = self.get_new_workflow()
            workflow_pp.workflow[0].close()
            workflow_pp.workflow[0] = self.reset_scan(train_flag=True)
            workflow_pp.print_ml_operator_costs()
            schemes_list.append(SCHEME_PP)
            microsoft_pp = MicrosoftPPDP(workflow=workflow_pp)
            pp_filters_pp, time_log_pp = microsoft_pp.multiple_pps_accuracy_allocation()
            pp_filters_list.append(pp_filters_pp)
            ml_filters_list.append(workflow_pp.get_ml_filters())
            time_logs_list.append(time_log_pp)
            scan_opt_output_num_list.append(workflow_pp.workflow[0].output_num)
            scan_pp, sink_pp = workflow_pp.get_scan_sink()
            scan_pp.close()
            sink_pp.close()
            workflow_pp.print_ml_operator_costs()
            print("finish optimizing PP ...")
            print()
        elif self.scheme == SCHEME_CORE:  # SCHEME_CORE
            workflow_core = self.get_new_workflow()
            workflow_core.workflow[0].close()
            workflow_core.workflow[0] = self.reset_scan(train_flag=True)
            workflow_core.print_ml_operator_costs()
            schemes_list.append(SCHEME_CORE)
            scheme_core_opt = FineGrainedAA(workflow=workflow_core)
            pp_filters_core, ml_filters_core, time_log_core = scheme_core_opt.reorder_optimization()
            pp_filters_list.append(pp_filters_core)
            ml_filters_list.append(ml_filters_core)
            time_logs_list.append(time_log_core)
            scan_opt_output_num_list.append(workflow_core.workflow[0].output_num)
            scan_core, sink_core = workflow_core.get_scan_sink()
            scan_core.close()
            sink_core.close()
            workflow_core.print_ml_operator_costs()
            print("finish optimizing CORE ...")
            print()
        elif self.scheme == SCHEME_COREa:  # SCHEME_COREa
            workflow_core_a = self.get_new_workflow()
            workflow_core_a.workflow[0].close()
            workflow_core_a.workflow[0] = self.reset_scan(train_flag=True)
            workflow_core_a.print_ml_operator_costs()
            schemes_list.append(SCHEME_COREa)
            plan_generator_core_a = PlanGenerator(workflow=workflow_core_a)
            pp_filters_core_a, time_log_core_a = plan_generator_core_a.one_plan_accuracy_allocation(
                plan=plan_generator_core_a.generate_each_pp_plan())
            pp_filters_list.append(pp_filters_core_a)
            ml_filters_list.append(workflow_core_a.get_ml_filters())
            time_logs_list.append(time_log_core_a)
            scan_opt_output_num_list.append(workflow_core_a.workflow[0].output_num)
            scan_core_a, sink_core_a = workflow_core_a.get_scan_sink()
            scan_core_a.close()
            sink_core_a.close()
            workflow_core_a.print_ml_operator_costs()
            print("finish optimizing COREa ...")
            print()
        elif self.scheme == SCHEME_COREh:  # SCHEME_COREh
            workflow_core_h = self.get_new_workflow()
            workflow_core_h.workflow[0].close()
            workflow_core_h.workflow[0] = self.reset_scan(train_flag=True)
            workflow_core_h.print_ml_operator_costs()
            schemes_list.append(SCHEME_COREh)
            scheme_core_h = ExhaustedSearch(workflow=workflow_core_h)
            pp_filters_core_h, ml_filters_core_h, time_log_core_h = scheme_core_h.reorder_optimization()
            pp_filters_list.append(pp_filters_core_h)
            ml_filters_list.append(ml_filters_core_h)
            time_logs_list.append(time_log_core_h)
            scan_opt_output_num_list.append(workflow_core_h.workflow[0].output_num)
            scan_core_h, sink_core_h = workflow_core_h.get_scan_sink()
            scan_core_h.close()
            sink_core_h.close()
            workflow_core_h.print_ml_operator_costs()
            print("finish optimizing COREh ...")
            print()
        else:  # SCHEME_REORDER
            workflow_reorder = self.get_new_workflow()
            workflow_reorder.workflow[0].close()
            workflow_reorder.workflow[0] = self.reset_scan(train_flag=True)
            workflow_reorder.print_ml_operator_costs()
            schemes_list.append(SCHEME_REORDER)
            scheme_reorder = ReorderBaseline(workflow=workflow_reorder)
            pp_filters_reorder, ml_filters_reorder, time_log_reorder = scheme_reorder.reorder_optimization()
            pp_filters_list.append(pp_filters_reorder)
            ml_filters_list.append(ml_filters_reorder)
            time_logs_list.append(time_log_reorder)
            scan_opt_output_num_list.append(workflow_reorder.workflow[0].operator_name)
            scan_reorder, sink_reorder = workflow_reorder.get_scan_sink()
            scan_reorder.close()
            sink_reorder.close()
            workflow_reorder.print_ml_operator_costs()
            print("finish optimizing REORDER ...")
            print()

        print("optimization details ...")
        for i in range(len(schemes_list)):
            print("scheme = " + str(schemes_list[i]) + "\ttime_log = " + str(
                time_logs_list[i]) + "\tscan_opt_output_num = " + str(scan_opt_output_num_list[i]))

        return schemes_list, pp_filters_list, ml_filters_list, time_logs_list, scan_opt_output_num_list

    def execution(self, schemes_list: List[int], pp_filters_list: List[List[PPFilter]],
                  ml_filters_list: List[List[MLFilter]]):
        print("start execution phase ...")
        accuracies_list, estimated_execution_costs_list = [], []
        ml_operators = self.get_ml_operators()

        all_input_records, original_output, original_cost = [], [], 0
        scan = self.reset_scan(train_flag=False)
        batch_records = scan.get_block_records(self.exe_batch_size)
        while batch_records:
            if scan.output_num % self.exe_batch_size == 0:
                print(scan.output_num)
            self.preprocessor.multiple_processes_batch(batch=batch_records, num_process=self.exe_thread_num)
            block_records = batch_records.copy()
            for ml_filter in ml_filters_list[0]:
                if self.workflow_type == TWITTER_TYPE:
                    ml_batch_output = ml_filter.multiple_threads_batch(batch=block_records,
                                                                       num_process=self.exe_thread_num,
                                                                       model=self.models[
                                                                           ml_filter.ml_predicator.ml_operator.operator_name])
                else:
                    ml_batch_output = ml_filter.multiple_processes_batch(batch=block_records,
                                                                         num_process=self.exe_thread_num)
                copy_filter_batch_out_result_all(large_batch_records=batch_records,
                                                 small_batch_records_has_results=block_records)
                block_records = ml_batch_output.returned_batch
                original_cost += ml_batch_output.processed_time
            batch_records_delete_ml_operator_results(batch_records=batch_records, ml_operators=ml_operators)
            all_input_records.extend(batch_records)
            original_output.extend(block_records)
            print("input num = " + str(len(all_input_records)) + "\toriginal output num = " + str(len(original_output)))

            batch_records = scan.get_block_records(self.exe_batch_size)
        original_input_num = len(all_input_records)
        original_output_num = len(original_output)

        estimated_execution_costs_list.append(original_cost / original_input_num)
        accuracies_list.append(1.0)
        scan.close()

        for i in range(1, len(schemes_list)):
            all_input_records_copy = all_input_records.copy()
            scheme_cost = 0
            print(schemes_list[i])
            """
            because we only parallel preprocessor. we did not parallel pp_inference phase.
            """
            for pp_filter in pp_filters_list[i]:
                scheme_cost += len(all_input_records_copy) * pp_filter.accuracy_info_dict.infer_cost
                pp_batch_output = pp_filter.multiple_threads_batch(batch=all_input_records_copy,
                                                                   num_process=self.exe_thread_num)
                # pp_batch_output = pp_filter.multiple_processes_batch(batch=all_input_records_copy,
                #                                                      num_process=self.exe_thread_num)
                all_input_records_copy = pp_batch_output.returned_batch

            for ml_filter in ml_filters_list[i]:
                if self.workflow_type == TWITTER_TYPE:
                    ml_batch_output = ml_filter.multiple_threads_batch(batch=all_input_records_copy,
                                                                       num_process=self.exe_thread_num,
                                                                       model=self.models[
                                                                           ml_filter.ml_predicator.ml_operator.operator_name])
                else:
                    ml_batch_output = ml_filter.multiple_processes_batch(batch=all_input_records_copy,
                                                                         num_process=self.exe_thread_num)
                all_input_records_copy = ml_batch_output.returned_batch
                scheme_cost += ml_batch_output.processed_time
            estimated_execution_costs_list.append(scheme_cost / original_input_num)
            accuracies_list.append(len(all_input_records_copy) / original_output_num)
            print("scheme = " + str(schemes_list[i]) + "\testimated execution cost = " + str(
                scheme_cost / original_input_num) + "\taccuracy = " + str(
                len(all_input_records_copy) / original_output_num))
        print("execution details ...")
        for i in range(len(schemes_list)):
            print("scheme = " + str(schemes_list[i]) + "\t" + "execution cost = " + str(
                estimated_execution_costs_list[i]) + "\taccuracy = " + str(accuracies_list[i]))
        return estimated_execution_costs_list, accuracies_list

    def run(self):
        self.set_operators_cost()
        schemes_list, pp_filters_list, ml_filters_list, time_logs_list, scan_opt_output_num_list = self.optimization()
        estimated_execution_costs_list, accuracies_list = self.execution(schemes_list=schemes_list,
                                                                         pp_filters_list=pp_filters_list,
                                                                         ml_filters_list=ml_filters_list)
        print("optimization time logs")
        for i in range(len(schemes_list)):
            print(str(schemes_list[i]) + ";\t" + str(time_logs_list[i]) + ";\t" + str(scan_opt_output_num_list[i]))
        print("execution cost and accuracy")
        for i in range(len(schemes_list)):
            print(
                str(schemes_list[i]) + ";\t" + str(estimated_execution_costs_list[i]) + ";\t" + str(accuracies_list[i]))
