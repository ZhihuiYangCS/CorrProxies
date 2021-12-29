import sys
import rootpath


rootpath.append()
from ml_workflow.workflow_base import WorkflowBase
from optimizer.accuracy_allocator.acc_allocator.accuracy_allocator_utility import generate_exhausted_search_space, \
    AccCombPPFiltersMap
from optimizer.compare_methods.microsoft_pp_base import MicrosoftPPBase


class MicrosoftPPES(MicrosoftPPBase):
    """
    implementing the method in the paper "Accelerating Machine Learning Inference with Probabilistic Predicates"
    """

    def __init__(self, workflow: WorkflowBase):
        super().__init__(workflow=workflow)

    def search(self):
        """
        exhausted search
        :return:  -> List[PPFilter]
        """
        self.ml_operator_costs = self.workflow.get_ml_operator_costs()
        print("ml_operator_costs = " + str(self.ml_operator_costs))
        search_space = generate_exhausted_search_space(accuracy_num=len(self.ml_filters),
                                                       target_accuracy=self.workflow.target_accuracy, eta=self.eta)
        result_acc_comb_pps = AccCombPPFiltersMap(accuracy_combination=[], cost=sys.maxsize, pp_filters=[],
                                                  retrain_flag=True)
        temp_index = 0
        for accuracy_combination in search_space:
            acc_comb_pps = self.one_accuracy_combination_cost(accuracy_combination=accuracy_combination)
            if acc_comb_pps.cost < result_acc_comb_pps.cost:
                result_acc_comb_pps = acc_comb_pps
            temp_index += 1
        print("final accuracy_combination = " + str(result_acc_comb_pps.accuracy_combination) + "final cost = " + str(
            result_acc_comb_pps.cost))
        print(" selected accuracy_reduction allocation = ")
        for i in range(len(result_acc_comb_pps.pp_filters)):
            print(result_acc_comb_pps.pp_filters[i].accuracy_info_dict)
        return result_acc_comb_pps.pp_filters

