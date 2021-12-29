from typing import List
import rootpath

rootpath.append()
from records.record import Record


def release_list_records(batch_records: List[Record]):
    del batch_records[:]
    del batch_records


def copy_filter_batch_out_result_all(large_batch_records: List[Record], small_batch_records_has_results: List[Record]):
    index_1, index_2 = 0, 0
    while True:
        if index_1 >= len(large_batch_records) or index_2 >= len(small_batch_records_has_results):
            break
        if large_batch_records[index_1].id == small_batch_records_has_results[index_2].id:
            large_batch_records[index_1] = small_batch_records_has_results[index_2]
            index_1 += 1
            index_2 += 1
        else:
            index_1 += 1
