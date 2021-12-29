import time
from typing import List
import rootpath

rootpath.append()
from utility.utility import sample_train_stop_condition, copy_samples, train_validate_test_split
from records.record import Record


class PlanSampleLabeler:
    """
    generate labeled samples for a passed plan, which indicates how to train PPs and train PPs on which ml_filters.
    """

    def __init__(self, workflow, plan: List[List[int]], workflow_samples: List[Record]):
        super().__init__()
        self.workflow = workflow
        self.plan = plan
        self.workflow_samples = workflow_samples
        # List[List[Record]]
        self.subplan_samples = []

    def get_subplan_samples(self):
        """
        get all labeled samples for subplans in plan,
        for example, we have a workflow A,B,C ---> we train pps on AB, C.
        This method get labeled samples for AB and C.
        """
        time0 = time.time()
        ml_filters = self.workflow.get_ml_filters()
        samples = self.workflow_samples
        for sub_plan in self.plan:
            labeled_sample = []
            pass_samples = []
            process_num = 0
            pass_num = 0
            labeled_sample_flag = False
            for record in samples:
                label = 1
                for index in sub_plan:
                    if ml_filters[index].ml_predicator.get_result(record) is not None:
                        label *= ml_filters[index].ml_predicator.get_result(record)
                    else:
                        label = 0
                        break
                if label:
                    pass_num += 1
                    pass_samples.append(record)
                process_num += 1
                if not labeled_sample_flag:
                    new_record = record.copy()
                    new_record["flag"] = label
                    labeled_sample.append(new_record)
                    labeled_sample_flag = sample_train_stop_condition(workflow_type=self.workflow.workflow_type,
                                                                      process_num=process_num, pos_num=pass_num)
            print("workflow_samples num = " + str(len(self.workflow_samples)) + "\tsubplan pass samples num = " + str(
                len(samples)) + "\tlabeled sample num = " + str(len(labeled_sample)) + "\tprocess_num = " + str(
                process_num) + " pos_num = " + str(pass_num) + " pass_rate = " + str(pass_num / process_num))
            labeled_sample = copy_samples(workflow_type=self.workflow.workflow_type, sample=labeled_sample)
            train_validate_test_split(labeled_sample)
            self.subplan_samples.append(labeled_sample)
            # samples = pass_samples.copy()
            samples = pass_samples
        time1 = time.time()
        print("plan sample labeler: cost = " + str(time1 - time0))
        # release_list_records(self.workflow_samples)
