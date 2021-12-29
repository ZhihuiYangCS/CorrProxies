from typing import List
import rootpath

rootpath.append()
from ml_workflow.workflow_base import WorkflowBase
from operators.ml_filter import MLFilter
from operators.operator_base.operator_parallel import OperatorParallel
from operators.scan.image_json_scan import ImageJsonScan
from operators.scan.text_json_scan import TextJsonScan
from operators.scan.video_json_scan import VideoJsonScan
from optimizer.PPs.pp_filter import PPFilter
from optimizer.PPs.preprocessor.preprocessor import Preprocessor
from optimizer.compare_methods.microsoft_pp_dynamic_programming import MicrosoftPPDP
from optimizer.plan_generator.plan_generator import PlanGenerator
from optimizer.reorder.reorder.tree.st_tree.fine_grained_accuracy_allocation import FineGrainedAA
from paths import TEXT_TWITTER_TRAIN_DATA_PATH, IMAGE_COCO_TRAIN_DATA_PATH, VIDEO_UCF101_TRAIN_DATA_PATH, \
    TEXT_TWITTER_VALI_DATA_PATH, IMAGE_COCO_VALI_DATA_PATH, VIDEO_UCF101_VALI_DATA_PATH
from utility.constant import TARGET_ACCURACY_DEFAULT, SCHEME_DEFAULT, OPT_THREAD_NUM_DEFAULT, EXE_THREAD_NUM_DEFAULT, \
    TWITTER_TYPE, SCHEME_NS, SCHEME_PP, target_accuracyies, TWITTER_EXE_BATCH_SIZE, COCO_TYPE, COCO_EXE_BATCH_SIZE, \
    UCF101_EXE_BATCH_SIZE
from utility.utility import batch_records_delete_ml_operator_results
from utility.utility_base import copy_filter_batch_out_result_all


class WorkflowExp4(WorkflowBase):
    """
    Different target_accuracy
    target_accuracy = 0.85, 0.9, 0.95
    schema == 0: accuracy allocation
    schema == others: COREs
    """

    def __init__(self, operators: List[OperatorParallel], preprocessor: Preprocessor, models: dict,
                 workflow_name: str = "", target_accuracy: float = TARGET_ACCURACY_DEFAULT,
                 scheme: int = SCHEME_DEFAULT, opt_thread_num: int = OPT_THREAD_NUM_DEFAULT,
                 exe_thread_num: int = EXE_THREAD_NUM_DEFAULT):
        super().__init__(operators=operators, preprocessor=preprocessor, models=models, workflow_name=workflow_name,
                         target_accuracy=target_accuracy, scheme=scheme, opt_thread_num=opt_thread_num)
        self.exe_thread_num = exe_thread_num
        if self.workflow_type == TWITTER_TYPE:
            self.exe_batch_size = TWITTER_EXE_BATCH_SIZE
        elif self.workflow_type == COCO_TYPE:
            self.exe_batch_size = COCO_EXE_BATCH_SIZE
        else:
            self.exe_batch_size = UCF101_EXE_BATCH_SIZE
        print("scheme = " + str(self.scheme))

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

    def set_operator_cost(self):
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
        pp_filters_list, ml_filters_list, time_logs_list, scan_opt_output_num_list = [], [], [], []
        if self.scheme == SCHEME_NS:  # SCHEME_NS
            for target_accuracy in target_accuracyies:
                print("schema = " + str(self.scheme) + "; start optimize target_accuracy = " + str(target_accuracy))
                workflow = self.get_new_workflow()
                workflow.workflow[0].close()
                workflow.workflow[0] = self.reset_scan(train_flag=True)
                workflow.print_ml_operator_costs()
                workflow.target_accuracy = target_accuracy

                plan_generator = PlanGenerator(workflow=workflow)
                pp_filters, time_log = plan_generator.one_plan_accuracy_allocation(
                    plan=plan_generator.generate_one_single_plan())

                pp_filters_list.append(pp_filters)
                ml_filters = workflow.get_ml_filters()
                ml_filters_list.append(ml_filters)
                time_logs_list.append(time_log)
                scan_opt_output_num = workflow.workflow[0].output_num
                scan_opt_output_num_list.append(scan_opt_output_num)

                scan, sink = workflow.get_scan_sink()
                scan.close()
                sink.close()
                workflow.print_ml_operator_costs()

                print("schema = " + str(self.scheme) + "; finish optimize target_accuracy = " + str(
                    target_accuracy) + "\t time_log = " + str(time_log) + "\t scan_opt_output_num = " + str(
                    scan_opt_output_num))
                print()
            print("finish optimizing NS ...")
            print()

        elif self.scheme == SCHEME_PP:  # SCHEME_PP
            for target_accuracy in target_accuracyies:
                print("schema = " + str(self.scheme) + "; start optimize target_accuracy = " + str(target_accuracy))
                workflow = self.get_new_workflow()
                workflow.workflow[0].close()
                workflow.workflow[0] = self.reset_scan(train_flag=True)
                workflow.print_ml_operator_costs()
                workflow.target_accuracy = target_accuracy

                scheme_method = MicrosoftPPDP(workflow=workflow)
                pp_filters, time_log = scheme_method.multiple_pps_accuracy_allocation()

                ml_filters = workflow.get_ml_filters()
                pp_filters_list.append(pp_filters)
                ml_filters_list.append(ml_filters)
                time_logs_list.append(time_log)
                scan_opt_output_num = workflow.workflow[0].output_num
                scan_opt_output_num_list.append(scan_opt_output_num)

                scan, sink = workflow.get_scan_sink()
                scan.close()
                sink.close()
                workflow.print_ml_operator_costs()

                print("schema = " + str(self.scheme) + "; finish optimize target_accuracy = " + str(
                    target_accuracy) + "\t time_log = " + str(time_log) + "\t scan_opt_output_num = " + str(
                    scan_opt_output_num))
                print()
            print("finish optimizing PP ...")
            print()

        else:  # SCHEME_CORE
            for target_accuracy in target_accuracyies:
                print("schema = " + str(self.scheme) + "; start optimize target_accuracy = " + str(target_accuracy))
                workflow = self.get_new_workflow()
                workflow.workflow[0].close()
                workflow.workflow[0] = self.reset_scan(train_flag=True)
                workflow.print_ml_operator_costs()
                workflow.target_accuracy = target_accuracy

                scheme_method = FineGrainedAA(workflow=workflow)
                pp_filters, ml_filters, time_log = scheme_method.reorder_optimization()

                pp_filters_list.append(pp_filters)
                ml_filters_list.append(ml_filters)
                time_logs_list.append(time_log)
                scan_opt_output_num = workflow.workflow[0].output_num
                scan_opt_output_num_list.append(scan_opt_output_num)

                scan, sink = workflow.get_scan_sink()
                scan.close()
                sink.close()
                workflow.print_ml_operator_costs()

                print("schema = " + str(self.scheme) + "; finish optimize target_accuracy = " + str(
                    target_accuracy) + "\t time_log = " + str(time_log) + "\t scan_opt_output_num = " + str(
                    scan_opt_output_num))
                print()
            print("finish optimizing CORE ...")
            print()

        return pp_filters_list, ml_filters_list, time_logs_list, scan_opt_output_num_list

    def execution(self, pp_filters_list: List[List[PPFilter]], ml_filters_list: List[List[MLFilter]]):
        print("start execution phase ...")
        accuracies_list, estimated_execution_costs_list = [], []
        ml_operators = self.get_ml_operators()

        all_input_records, original_output, original_cost = [], [], 0
        scan = self.reset_scan(train_flag=False)
        batch_records = scan.get_block_records(self.exe_batch_size)
        while batch_records:
            if scan.output_num % self.exe_batch_size == 0:
                print(scan.output_num)
            batch_output = self.preprocessor.multiple_processes_batch(batch=batch_records, num_process=self.exe_thread_num)
            batch_records = batch_output.returned_batch
            block_records = batch_records.copy()
            for ml_filter in self.get_ml_filters():
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

        # estimated_execution_costs_list.append(original_cost / original_input_num)
        # accuracies_list.append(1.0)
        # scan.close()

        for i in range(len(target_accuracyies)):
            all_input_records_copy = all_input_records.copy()
            scheme_cost = 0
            print(target_accuracyies[i])
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
            print("target_accuracy = " + str(target_accuracyies[i]) + "\testimated execution cost = " + str(
                scheme_cost / original_input_num) + "\taccuracy = " + str(
                len(all_input_records_copy) / original_output_num))
        print("execution details ...")
        print("original\t" + str(original_cost/original_input_num) + "\t" + str(1.0))
        for i in range(len(target_accuracyies)):
            print(str(target_accuracyies[i]) + "\t" + str(
                estimated_execution_costs_list[i]) + "\t" + str(accuracies_list[i]))
        return estimated_execution_costs_list, accuracies_list, original_cost/original_input_num, 1.0

    def run(self):
        self.set_operator_cost()
        pp_filters_list, ml_filters_list, time_logs_list, scan_opt_output_num_list = self.optimization()
        estimated_execution_costs_list, accuracies_list, original_execution_cost, original_accuracy = self.execution(pp_filters_list=pp_filters_list, ml_filters_list=ml_filters_list)
        print("time_log results ..................")
        for i in range(len(target_accuracyies)):
            print(str(target_accuracyies[i]) + "\t" + str(time_logs_list[i]))
        print("scan_opt_output_num results ..................")
        for i in range(len(target_accuracyies)):
            print(str(target_accuracyies[i]) + "\t" + str(scan_opt_output_num_list[i]))
        print("original\t" + str(original_execution_cost) + "\t" + str(original_accuracy))
        for i in range(len(target_accuracyies)):
            print(str(target_accuracyies[i]) + "\t" + str(
                estimated_execution_costs_list[i]) + "\t" + str(accuracies_list[i]))
