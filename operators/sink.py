import json
import os
import time
from typing import Any, List

import rootpath

rootpath.append()
from operators.operator_base.operator_action import OperatorDynamicAction
from operators.scan.text_json_scan import TextJsonScan
from paths import TEMP_TEST_PATH, TEXT_TWITTER_TRAIN_DATA_PATH
from records.record import Record


class Sink(OperatorDynamicAction):
    """
    write results into a file
    """

    def __init__(self, write_path: str):
        super().__init__(os.path.basename(write_path))
        self.write_path = write_path
        self._file_writer = open(self.write_path + ".txt", "w")

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

    def write(self, records: List[Record]):
        """
        write a batch into file
        :param records: a list of records
        """
        for record in records:
            self._file_writer.write(json.dumps(record.id) + '\n')
        self.process_batch(records)

    def operator_reset(self):
        self.write_path = 0
        self.process_count = 0
        self.pass_count = 0
        self.process_time = 0
        self.close()
        self._file_writer = open(self.write_path + ".txt", "w")

    def get_new_object(self, operator_name: str = ""):
        """
        return a new object
        :param previous_operators: List[Operator]
        :param operator_name: str, default value is ""
        """
        return Sink(write_path=self.write_path + "_copy")

    def close(self):
        self._file_writer.close()


"""
test units
"""


def sink_operator_name_test():
    sink = Sink(TEMP_TEST_PATH)
    print(sink)
    sink.close()


def sink_write_test():
    """
    sink write test
    """
    sink = Sink(TEMP_TEST_PATH)
    scan = TextJsonScan(TEXT_TWITTER_TRAIN_DATA_PATH)
    block_records = scan.get_block_records(10)
    sink.write(block_records)
    scan.close()
    sink.close()


def operator_copy_test():
    time0 = time.time()
    operator = Sink(TEMP_TEST_PATH)
    time1 = time.time()
    operator_copy = operator.get_new_object()
    print("create model time = " + str(time1 - time0) + "\tcopy model time = " + str(time.time() - time1))

    print("operator.write_path = " + str(operator.write_path))
    print("operator_copy.write_path = " + str(operator_copy.write_path))
    operator.close()
    operator_copy.close()


if __name__ == '__main__':
    pass
#     sink_operator_name_test()
#     sink_write_test()
#     operator_copy_test()
