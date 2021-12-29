import time
from abc import ABC, abstractmethod
from typing import List

import rootpath

rootpath.append()
from utility.constant import AA_MODE_S_T, AA_MODE_RS_RT
from optimizer.accuracy_allocator.acc_allocator.accuracy_allocator_utility import AccCombPPFiltersMap
from optimizer.accuracy_allocator.pp_manager_base import PPManagerBase
from optimizer.accuracy_allocator.plan_sample_labeler import PlanSampleLabeler
from optimizer.plan_generator.workflow_labeled_sample import WorkflowLabeledSample
from records.record import Record


class AccuracyAllocatorBase(ABC):
    """
    allocation_mode: 0 ---- resample retrain
                     1 ---- reuse sample, retrain
                     2 ---- reuse sample, reuse PPs
    """

    def __init__(self, workflow, plan: List[List[int]], workflow_samples: List[Record] = None):
        self.workflow = workflow
        self.plan = plan
        self.eta = 0.99
        self.workflow_samples = workflow_samples
        self.plan_sample_labeler = None
        # List[PPManager]
        self.plan_pp_managers = []
        # List[MLFilter]
        self.ml_filters = self.workflow.get_ml_filters()
        # List[List[float]]
        self.search_space = []
        self.label_time = 0
        self.train_time = 0
        self.aa_time = 0
        self.reorder_time = 0
        print("target_accuracy = " + str(self.workflow.target_accuracy))

    def _get_samples(self):
        """
        if allocation_mode == AA_MODE_RS_RT (resample) or workflow_samples is None:
            generate workflow_level samples;
            get plan level samples;
        if plan_sample_labeler is None: get plan level samples.
        """
        if self.workflow.allocation_mode == AA_MODE_RS_RT or self.workflow_samples is None:
            time1 = time.time()
            workflow_labeled_sample = WorkflowLabeledSample(workflow=self.workflow)
            self.workflow_samples = workflow_labeled_sample.generate_workflow_samples()
            time2 = time.time()
            self.label_time += (time2 - time1)
            self.plan_sample_labeler = PlanSampleLabeler(workflow=self.workflow, plan=self.plan,
                                                         workflow_samples=self.workflow_samples)
            self.plan_sample_labeler.get_subplan_samples()

        if self.plan_sample_labeler is None:
            self.plan_sample_labeler = PlanSampleLabeler(workflow=self.workflow, plan=self.plan,
                                                         workflow_samples=self.workflow_samples)
            self.plan_sample_labeler.get_subplan_samples()

    def initialize_plan_pp_managers(self):
        """
        initialize self.plan_pps.
        """
        for i in range(len(self.plan)):
            if self.workflow.allocation_mode == AA_MODE_S_T:  # reuse samples and reuse PPs
                pp_manager = PPManagerBase(workflow=self.workflow)
            else:
                pp_manager = PPManagerBase(workflow=self.workflow,
                                           reuse_flag=False)
            self.plan_pp_managers.append(pp_manager)

    @abstractmethod
    def one_accuracy_combination_cost(self, accuracy_combination: List[float]) -> AccCombPPFiltersMap:
        """
        compute the cost for one accuracy combination
        :param accuracy_combination: the accuracy combination that we compute the cost
        :return: an accuracy_combination_pp_filters_map
        """
        pass

    @abstractmethod
    def search(self):
        """
        an abstract method, implement exhausted search and provide an outer interface to perform accuracy allocation
        :return: a list of PPFilter trained on the search accuracy combination which has a minial cost.
        """
        pass
