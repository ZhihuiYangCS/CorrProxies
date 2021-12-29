import time
from typing import Any

import rootpath

rootpath.append()
from paths import TEXT_TWITTER_TRAIN_DATA_PATH
from records.record import Record
from operators.scan.text_json_scan import TextJsonScan
from operators.ml_operators.text_operators.stanfordnlp.stanfordnlp_operator import StanfordNLPOperator


class SentimentStanfordNLP(StanfordNLPOperator):
    def __init__(self):
        super().__init__()
        self.properties = {"annotators": "tokenize, ssplit, parse, sentiment"}

    def get_operator_cost(self):
        """return operator time cost per record, in ms/record
            :raises ValueError
        """
        if SentimentStanfordNLP.operator_cost is None:
            if self.process_count == 0:
                print("Getting operator cost: process_count = " + str(
                    self.process_count) + "\tYou may forget add processing statistic informations ...")
                SentimentStanfordNLP.operator_cost = None
            else:
                SentimentStanfordNLP.operator_cost = self.process_time / self.process_count

    def _process(self, one: Record, model=None) -> Any:
        k = super()._process(one, model=model)
        if k is None:
            return ''
        else:
            return [s for s in k['sentences']]

    def get_new_object(self, previous_operators=None, operator_name: str = ""):
        """
        return a new object
        :param previous_operators: List[Operator]
        :param operator_name: str, default value is ""
        """
        return SentimentStanfordNLP()


"""
test units
"""


def operator_name_test():
    operator = SentimentStanfordNLP()
    model = operator.load_model()
    print(operator)
    model.close()


def operator_test():
    """
    operator block process test
    """
    scan = TextJsonScan(TEXT_TWITTER_TRAIN_DATA_PATH)
    block_records = scan.get_block_records(10)
    operator = SentimentStanfordNLP()
    model = operator.load_model()
    batch_output = operator.process_batch(batch=block_records, model=model)
    for record in batch_output.returned_batch:
        print(str(operator.get_result(record)).encode('utf-8'))
    print(batch_output)
    scan.close()
    model.close()


def operator_copy_test():
    time0 = time.time()
    operator = SentimentStanfordNLP()
    time1 = time.time()
    operator_copy = operator.get_new_object()
    print("create model time = " + str(time1 - time0) + "\tcopy model time = " + str(time.time() - time1))

    print("operator.process_count = " + str(operator.process_count))
    print("operator_copy.process_count = " + str(operator_copy.process_count))

    operator.process_count = 1000
    print("operator.process_count = " + str(operator.process_count))
    print("operator_copy.process_count = " + str(operator_copy.process_count))
    operator_copy.process_count = 2000
    print("operator.process_count = " + str(operator.process_count))
    print("operator_copy.process_count = " + str(operator_copy.process_count))
    operator.operator_reset()
    print("operator.process_count = " + str(operator.process_count))
    print("operator_copy.process_count = " + str(operator_copy.process_count))
    operator_copy.operator_reset()
    print("operator.process_count = " + str(operator.process_count))
    print("operator_copy.process_count = " + str(operator_copy.process_count))


def operator_cost_test():
    time1 = time.time()
    scan = TextJsonScan(TEXT_TWITTER_TRAIN_DATA_PATH)
    block_records = scan.get_block_records(100)
    operator = SentimentStanfordNLP()
    model = operator.load_model()
    batch_output = operator.process_batch(batch=block_records, model=model)
    operator.get_operator_cost()
    print("operator cost = " + str(operator.operator_cost) + "\t " + str(batch_output) + "\n")
    time2 = time.time()

    block_records = scan.get_block_records(100)
    batch_output = operator.process_batch(batch=block_records, model=model)
    operator.get_operator_cost()
    print("operator cost = " + str(operator.operator_cost) + "\t " + str(batch_output) + "\n")

    operator_new = SentimentStanfordNLP()
    print("new operator cost = " + str(operator_new.operator_cost) + "\n")

    scan.close()
    model.close()
    time3 = time.time()
    print("one process: one block time = " + str(time2 - time1) + "\t two blocks time = " + str(time3 - time2) + "\n\n")


def multple_threads_test():
    time1 = time.time()
    scan = TextJsonScan(TEXT_TWITTER_TRAIN_DATA_PATH)
    block_records = scan.get_block_records(100)
    operator = SentimentStanfordNLP()
    model = operator.load_model()
    batch_output = operator.multiple_threads_batch(batch=block_records, num_process=10, model=model)
    operator.get_operator_cost()
    print("operator cost = " + str(operator.operator_cost) + "\t " + str(batch_output) + "\n")
    time2 = time.time()

    block_records = scan.get_block_records(100)
    batch_output = operator.multiple_threads_batch(batch=block_records, num_process=10, model=model)
    operator.get_operator_cost()
    print("operator cost = " + str(operator.operator_cost) + "\t " + str(batch_output) + "\n")

    operator_new = SentimentStanfordNLP()
    print("new operator cost = " + str(operator_new.operator_cost) + "\n")

    scan.close()
    model.close()
    time3 = time.time()
    print("one process: one block time = " + str(time2 - time1) + "\t two blocks time = " + str(time3 - time2) + "\n\n")


def multple_processes_test():
    time1 = time.time()
    scan = TextJsonScan(TEXT_TWITTER_TRAIN_DATA_PATH)
    block_records = scan.get_block_records(100)
    operator = SentimentStanfordNLP()
    batch_output = operator.multiple_processes_batch(batch=block_records, num_process=10)
    operator.get_operator_cost()
    print("operator cost = " + str(operator.operator_cost) + "\t " + str(batch_output) + "\n")
    time2 = time.time()

    block_records = scan.get_block_records(100)
    batch_output = operator.multiple_processes_batch(batch=block_records, num_process=10)
    operator.get_operator_cost()
    print("operator cost = " + str(operator.operator_cost) + "\t " + str(batch_output) + "\n")

    operator_new = SentimentStanfordNLP()
    print("new operator cost = " + str(operator_new.operator_cost) + "\n")

    scan.close()
    time3 = time.time()
    print("multiple process: one block time = " + str(time2 - time1) + "\t two blocks time = " + str(
        time3 - time2) + "\n")


if __name__ == '__main__':
    pass
#     operator_name_test()
#     operator_test()
#     operator_copy_test()
#     operator_cost_test()
#     multple_threads_test()
#     multple_processes_test()
