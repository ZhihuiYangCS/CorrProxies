import json
import time
from typing import Union

import rootpath

rootpath.append()
from operators.scan.json_scan import JsonScan
from paths import TEXT_TWITTER_TRAIN_DATA_PATH
from records.record import Record


class TextJsonScan(JsonScan):
    """
    An interface to scan all text json data
    """

    def __init__(self, read_path: str, file_mode=False):
        super().__init__(read_path=read_path, file_mode=file_mode)

    def _read_one_record_from_file(self) -> Union[Record, None]:
        """
        read one record from file
        :return: a record or None. None means the end of file
        """
        record: Union[Record, None] = None
        if not self._file_reader.closed:
            line = self._file_reader.readline()
            while line.strip():
                if line.startswith("{"):
                    json_data = json.loads(line)
                    record = Record(id=int(json_data["id"]), value=json_data["text"])
                    self.output_num += 1
                    break
                else:
                    line = self._file_reader.readline()
        return record

    def get_new_object(self, previous_operators=None, operator_name: str = ""):
        """
        return a new object
        :param previous_operators: List[Operator]
        :param operator_name: str, default value is ""
        """
        return TextJsonScan(read_path=self._read_path, file_mode=self.file_mode)


"""
test units
"""


def scan_name_test():
    """
    test the json_scan operator name
    :return:
    """
    scan = TextJsonScan(TEXT_TWITTER_TRAIN_DATA_PATH)
    print(scan)


def scan_one_record_test():
    """
    a test method to test scan read one record
    """
    scan = TextJsonScan(TEXT_TWITTER_TRAIN_DATA_PATH)
    records_num = 1
    while records_num <= 10:
        record = scan.get_one_record()
        print(str(record).encode('utf-8'))
        records_num += 1
    print("we need to gets 10 records, we have scanned " + str(scan.output_num))
    scan.close()


def scan_block_test():
    """
        a test method to test scan read a block of records
    """
    scan = TextJsonScan(TEXT_TWITTER_TRAIN_DATA_PATH)
    block_num = 1
    while block_num < 3:
        print("the " + str(block_num) + " block")
        block_records = scan.get_block_records(5)
        print(str(block_records).encode('utf-8'))
        block_num += 1
    print("we need to get 10 records, we have scanned " + str(scan.output_num))
    scan.close()


def scan_test():
    """
    scan a directory. usage test
    """
    scan = TextJsonScan(TEXT_TWITTER_TRAIN_DATA_PATH)
    block_records = scan.get_block_records(1000)
    while block_records:
        if scan.output_num % 1000 == 0:
            print(str(scan.output_num))
        block_records = scan.get_block_records(1000)
    scan.close()


def scan_copy_test():
    time0 = time.time()
    scan = TextJsonScan(TEXT_TWITTER_TRAIN_DATA_PATH)
    time1 = time.time()
    scan_copy = scan.get_new_object()
    print("create model time = " + str(time1 - time0) + "\tcopy model time = " + str(time.time() - time1))
    print("scan.output_num = " + str(scan.output_num))
    print("scan_copy.output_num = " + str(scan_copy.output_num))
    scan.output_num = 1
    print("scan.output_num = " + str(scan.output_num))
    print("scan_copy.output_num = " + str(scan_copy.output_num))
    scan_copy.output_num = 2
    print("scan.output_num = " + str(scan.output_num))
    print("scan_copy.output_num = " + str(scan_copy.output_num))
    scan.operator_reset()
    print("scan.output_num = " + str(scan.output_num))
    print("scan_copy.output_num = " + str(scan_copy.output_num))
    scan_copy.operator_reset()
    print("scan.output_num = " + str(scan.output_num))
    print("scan_copy.output_num = " + str(scan_copy.output_num))
    scan.close()
    scan_copy.close()


def get_interval_block_records_test():
    scan = TextJsonScan(TEXT_TWITTER_TRAIN_DATA_PATH)
    print("get one record")
    print(str(scan.get_one_record()).encode('utf-8'))
    print(str(scan.get_one_record()).encode('utf-8'))
    print("get interval block records")
    scan_copy = scan.get_new_object()
    records = scan_copy.get_interval_block_records(start_index=2, end_index=3, batch_size=1)
    while records:
        for record in records:
            print(str(record).encode('utf-8'))
        records = scan_copy.get_interval_block_records(start_index=2, end_index=3, batch_size=1)


if __name__ == '__main__':
    pass
#     scan_name_test()
#     print("scan one record test")
#     scan_one_record_test()
#     print("scan block test")
#     scan_block_test()
#     scan_test()
#     scan_copy_test()
#     get_interval_block_records_test()
