import glob
from abc import ABC, abstractmethod
from typing import Any, List, Union

import rootpath

rootpath.append()

from operators.scan.scan import Scan
from records.record import Record


class JsonScan(Scan, ABC):
    """
    read a directory of json files.
    get_one_record()
    get_block_records()
    reset_to_beginning()
    close()
    """

    def __init__(self, read_path: str, file_mode=False):
        # super().__init__(os.path.basename(read_path))
        super().__init__()
        self._read_path = read_path
        self.file_mode = file_mode
        if self.file_mode:
            self._files = [read_path]
        else:
            self._files = glob.glob(self._read_path + "/*")
        self._folder_size = len(self._files)
        self.output_num = 0
        if self._folder_size < 1:
            raise FileNotFoundError("Error: the folder " + self._read_path + " is empty")
        else:
            self._file_index = 0
            self._file_reader = open(self._files[0], "r")

    @abstractmethod
    def _read_one_record_from_file(self) -> Union[Record, None]:
        pass

    def get_one_record(self) -> Union[Record, None]:
        """
        get on record
        :return: a Record or None, None means the end of the input directory
        """
        record = self._read_one_record_from_file()
        if record is None:
            self._file_reader.close()
            self._file_index += 1
            while self._file_index < self._folder_size:
                self._file_reader = open(self._files[self._file_index], "r")
                record = self._read_one_record_from_file()
                if record is None:
                    self._file_reader.close()
                    self._file_index += 1
                else:
                    break
        # if self.output_num % 1000 == 0:
        #     print("scanned", self.output_num)
        return record

    def get_block_records(self, block_size: int) -> Union[List[Record]]:
        """
        read a block of records
        :param block_size: int
        :return: a list of records or None.
        """
        records = []
        record = self.get_one_record()
        while record:
            records.append(record)
            if len(records) >= block_size:
                break
            record = self.get_one_record()
        return records

    def get_interval_block_records(self, start_index: int, end_index: int, batch_size: int) -> Union[List[Record]]:
        """
        the interval is [a,b] including a and b, not (a,b)
        """
        records = []
        record = self.get_one_record()
        while self.output_num < start_index:
            record = self.get_one_record()
        while self.output_num <= end_index and record is not None:
            records.append(record)
            if len(records) >= batch_size:
                break
            record = self.get_one_record()
        return records

    def reset_to_beginning(self):
        """
        reset the _file_reader to the beginning
        """
        self.output_num = 0
        self._file_index = 0
        self._file_reader = open(self._files[0], "r")

    def operator_reset(self):
        """
        reset the _file_reader to the beginning
        """
        self.process_count = 0
        self.pass_count = 0
        self.process_time = 0

        self.output_num = 0
        self._file_index = 0
        self._file_reader = open(self._files[0], "r")

    def __str__(self):
        return self.operator_name

    def close(self):
        """
        close _file_reader
        """
        self._file_reader.close()
