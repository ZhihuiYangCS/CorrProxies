from abc import ABC, abstractmethod
from typing import Any, Union

import rootpath

rootpath.append()
from records.record import Record
from operators.operator_base.operator_utility import BatchOutput


class OperatorStaticBase(ABC):
    """
    basic Operator Static class, including process_count, pass_count, process_time, set_result, get_key etc.
    each operator has an operator_name
    """
    operator_cost: Union[None, float] = None

    def __init__(self, operator_name: str = ""):
        # def __init__(self, operator_name: str):
        self.operator_name = self.__class__.__name__ + (("_" + operator_name) if operator_name else "")
        self.process_count = 0
        self.pass_count = 0
        self.process_time = 0

    @abstractmethod
    def get_operator_cost(self):
        """
        a method to get operator's cost;
        for ML_Operator and Preprocessor, they are class level;
        for ML_Filter and ML_Predicator, they are instance level.
        """
        pass

    def get_operator_pass_rate(self):
        """
        compute operator's pass rate, and return the pass_rate
        """
        if self.process_count == 0:
            print("Getting pass rate: process_count = " + str(
                self.process_count) + "\tYou may forget add processing statistic informations ...")
            pass_rate = None
        else:
            pass_rate = self.pass_count / self.process_count
        return pass_rate

    def set_result(self, one: Record, result: Any):
        """
        add operator inference result to one record,
        the type of result can is any.
        It can be a list, str, dict ...
        """
        one[self.operator_name + '_RESULT'] = result

    def get_result_key(self) -> str:
        """
        get operator result key
        :return: str
        """
        return self.operator_name + '_RESULT'

    def get_result(self, one: Record) -> Any:
        """
        return operator inference result
        """
        return one.get_value(self.operator_name + '_RESULT')

    def set_processed_statistics(self, batch_output: BatchOutput):
        """
        set this operator's process_count, pass_count, and process_time
        """
        self.process_count += batch_output.processed_count
        self.pass_count += batch_output.passed_count
        self.process_time += batch_output.processed_time

    def operator_reset(self):
        """
        reset the operator status to initial state.
        """
        self.process_count = 0
        self.process_time = 0
        self.pass_count = 0

    def print_operator_detail(self):
        """
        an detailed print information for an operator
        """
        print("operator = " + self.operator_name + "\t operator_cost = " + str(
            self.operator_cost) + "\t process_count = " + str(self.process_count) + "\t pass_count = " + str(
            self.pass_count) + "\t process_time = " + str(self.process_time))

    def __str__(self) -> str:
        """
        string representation of operator
        """
        return self.operator_name

    @abstractmethod
    def get_new_object(self, previous_operators=None, operator_name: str = ""):
        """
        return a new object
        :param previous_operators: List[Operator]
        :param operator_name: str, default value is ""
        """
        pass

    __repr__ = __str__


if __name__ == '__main__':
    pass
