from abc import ABC, abstractmethod
from typing import List, Union, Any
import rootpath

rootpath.append()
from operators.operator_base.operator_base import OperatorStaticBase

from records.record import Record


class Scan(OperatorStaticBase, ABC):
    """
    scan base class
    """

    def get_operator_cost(self):
        pass

    def _process(self, one: Record, model=None) -> Any:
        """base method to get result of one record"""
        pass

    def load_model(self) -> Any:
        """
        loading an ML model, which is used for process one record
        """
        pass

    @abstractmethod
    def get_one_record(self) -> Union[Record, None]:
        """
        get one record
        :return: one Record or None
        """
        pass

    @abstractmethod
    def get_block_records(self, block_size: int) -> Union[List[Record]]:
        """
        get a block of records
        :param block_size: the size of one block
        :return: a list of records or None
        """
        pass

    def next(self):
        return self.get_one_record()
