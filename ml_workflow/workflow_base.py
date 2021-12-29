from abc import ABC
from typing import List

import rootpath

from operators.ml_operators.image_video_operators.video_activity_recognition import VideoActivityRecognition

rootpath.append()
from utility.constant import AA_METHOD_DEFAULT, AA_MODE_DEFAULT, REORDER_AA_METHOD_DEFAULT, \
    TARGET_ACCURACY_DEFAULT, SCHEME_DEFAULT, OPT_THREAD_NUM_DEFAULT, TWITTER_TYPE, TWITTER_PP_TYPE, COCO_TYPE, \
    COCO_PP_TYPE, UCF101_TYPE, UCF101_PP_TYPE, TWITTER_BATCH_SIZE, COCO_BATCH_SIZE, UCF101_BATCH_SIZE
from operators.scan.image_json_scan import ImageJsonScan
from operators.scan.text_json_scan import TextJsonScan
from operators.ml_filter import MLFilter
from operators.sink import Sink
from operators.scan.scan import Scan
from optimizer.PPs.preprocessor.preprocessor import Preprocessor
from operators.operator_base.operator_parallel import OperatorParallel


class WorkflowBase(ABC):
    """
    This class is an abstract of all static attributes of a workflow,
        which contains all operators in a workflow,
              get methods of operators,
              and check methods of a workflow
    """

    def __init__(self, operators: List[OperatorParallel], preprocessor: Preprocessor, models: dict,
                 workflow_name: str = "", target_accuracy: float = TARGET_ACCURACY_DEFAULT,
                 scheme: int = SCHEME_DEFAULT, opt_thread_num: int = OPT_THREAD_NUM_DEFAULT,
                 allocation_method: int = AA_METHOD_DEFAULT, allocation_mode: int = AA_MODE_DEFAULT,
                 reorder_aa_method: int = REORDER_AA_METHOD_DEFAULT):
        if workflow_name:
            self.workflow_name = workflow_name
        else:
            self.workflow_name = self.__class__.__name__
        self.workflow = operators
        self.preprocessor = preprocessor
        self.models = models
        self.scheme = scheme
        self.opt_thread_num = opt_thread_num
        if isinstance(self.workflow[0], TextJsonScan):
            self.workflow_type = TWITTER_TYPE
            self.batch_size = TWITTER_BATCH_SIZE
            self.pp_type = TWITTER_PP_TYPE
        elif isinstance(self.workflow[0], ImageJsonScan):
            self.workflow_type = COCO_TYPE
            self.batch_size = COCO_BATCH_SIZE
            self.pp_type = COCO_PP_TYPE
        else:
            self.workflow_type = UCF101_TYPE
            self.batch_size = UCF101_BATCH_SIZE
            self.pp_type = UCF101_PP_TYPE

        self.allocation_method = allocation_method
        self.allocation_mode = allocation_mode
        self.reorder_accuracy_allocator_method = reorder_aa_method
        self.target_accuracy = target_accuracy

        self.ml_filters = None
        self.ml_filters_operator_names = None
        self.ml_operators = None

        if self.workflow_type == UCF101_TYPE and self.opt_thread_num == 0:
            for ml_operator in self.get_ml_operators():
                self.models[ml_operator.operator_name] = ml_operator.load_model()
            self.models[self.preprocessor.operator_name] = self.preprocessor.load_model()

    def append_operator(self, operator: OperatorParallel):
        """
        append one operator into the workflow
        """
        self.workflow.append(operator)

    def get_scan_sink(self) -> [Scan, Sink]:
        """
        get the scan and sink operators of the workflow
        """
        scan, sink = None, None
        if isinstance(self.workflow[0], Scan):
            scan = self.workflow[0]
        else:
            self._exit_exception("This workflow does not have Scan operator")
        if isinstance(self.workflow[-1], Sink):
            sink = self.workflow[-1]
        else:
            self._exit_exception("This workflow does not have Sink operator")
        return scan, sink

    def get_ml_filters(self) -> List[MLFilter]:
        """
        get all MLFilters in the workflow
        :return: a list of MLFilters
        """
        if self.ml_filters is None:
            ml_filters = []
            for operator in self.workflow[1:-1]:
                if isinstance(operator, MLFilter):
                    ml_filters.append(operator)
                else:
                    self._exit_exception(operator.operator_name + " is not an MLFilter")
            self.ml_filters = ml_filters
        return self.ml_filters

    def get_ml_filters_operator_names(self) -> List[str]:
        """
        get all ml_filters' operator names
        :return:
        """
        if self.ml_filters_operator_names is None:
            self.ml_filters_operator_names = []
            ml_filters = self.get_ml_filters()
            for ml_filter in ml_filters:
                self.ml_filters_operator_names.append(ml_filter.operator_name)
        return self.ml_filters_operator_names

    def get_ml_operator_costs(self):
        """
        get all ml_operators costs in the workflow
        :return: a dict with key is the name of the ml_operator and value is the cost the ml_operator
        """
        ml_filters = self.get_ml_filters()
        for ml_filter in ml_filters:
            ml_filter.get_operator_cost()
            ml_filter.print_operator_detail()

    def print_ml_operator_costs(self):
        """
        print all ml_filters operator_costs
        """
        ml_filters = self.get_ml_filters()
        for ml_filter in ml_filters:
            print(ml_filter.operator_name + "\t operator_cost = " + str(ml_filter.operator_cost))

    def get_ml_operators(self) -> List[OperatorParallel]:
        """
        get all ml operators in the workflow
        :return: a list of ml operators
        """
        if self.ml_operators is None:
            ml_filters = self.get_ml_filters()
            ml_operators = set()
            for ml_filter in ml_filters:
                ml_operators.add(ml_filter.ml_predicator.ml_operator)
            self.ml_operators = list(ml_operators)
        return self.ml_operators

    def get_new_workflow(self):
        """
        create a new workflow
        :return: a new workflow
        """
        operators = []
        scan, sink = self.get_scan_sink()
        operators.append(scan.get_new_object())
        ml_filters = self.get_ml_filters()
        for index, ml_filter in enumerate(ml_filters):
            new_ml_filter = ml_filter.get_new_object(previous_operators=ml_filters[0:index],
                                                     operator_name=ml_filter.operator_name.split("_")[-1])
            if isinstance(ml_filter.ml_predicator.ml_operator, VideoActivityRecognition):
                new_ml_filter.ml_predicator.ml_operator = VideoActivityRecognition()
            operators.append(new_ml_filter)
        operators.append(sink.get_new_object())
        workflow = WorkflowBase(operators, preprocessor=self.preprocessor, models=self.models,
                                workflow_name=self.workflow_name + "_copy", target_accuracy=self.target_accuracy,
                                scheme=self.scheme, opt_thread_num=self.opt_thread_num,
                                allocation_method=self.allocation_method, allocation_mode=self.allocation_mode,
                                reorder_aa_method=self.reorder_accuracy_allocator_method)
        return workflow

    def workflow_check(self):
        """
        check if the workflow has one Scan at the front and one Sink at the end, and MLFilters in the middle
        """
        if not isinstance(self.workflow[0], Scan):
            self._exit_exception("This workflow does not have Scan operator")
        if not isinstance(self.workflow[-1], Sink):
            self._exit_exception("This workflow does not have Sink operator")
        for operator in self.workflow[1:-1]:
            if not isinstance(operator, MLFilter):
                self._exit_exception(operator.operator_name + " is not a MLFilter")

    def _exit_exception(self, message: str):
        """
        exit with exception message
        :param message: error message
        """
        print(message)
        exit(1)

    def __str__(self):
        return f"{self.workflow_name}[" + ' -> '.join(map(str, self.workflow)) + "]"

    def __getitem__(self, index):
        return self.workflow[index]

    def __len__(self):
        return len(self.workflow)
