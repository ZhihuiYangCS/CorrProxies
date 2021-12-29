import time
from typing import Any

import rootpath

rootpath.append()

from operators.scan.text_json_scan import TextJsonScan
from paths import TEXT_TWITTER_TRAIN_DATA_PATH
from records.record import Record
from operators.ml_operators.text_operators.stanfordnlp.stanfordnlp_operator import StanfordNLPOperator

"""
HANDLE
STATE_OR_PROVINCE, LOCATION, CITY, COUNTRY, pass_rate = 0.1198, cost = 0.01825
MONEY
NUMBER, PERCENT, ORDINAL, SET, pass_rate = 0.1292222222222222
NATIONALITY
PERSON, TITLE
EMAIL
URL
DATE, TIME, DURATION, ORDINAL, SET, pass_rate = 0.1691111111111111
MISC     not 0.8406666666666667
IDEOLOGY, CRIMINAL_CHARGE, CAUSE_OF_DEATH, RELIGION
ORGANIZATION
"""


class EntityStanfordNLP(StanfordNLPOperator):
    def __init__(self):
        super().__init__()
        self.properties = {"annotators": "tokenize, ssplit, pos, lemma, ner"}

    def get_operator_cost(self):
        """return operator time cost per record, in ms/record
            :raises ValueError
        """
        if EntityStanfordNLP.operator_cost is None:
            if self.process_count == 0:
                print("Getting operator cost: process_count = " + str(
                    self.process_count) + "\tYou may forget add processing statistic informations ...")
                EntityStanfordNLP.operator_cost = None
            else:
                EntityStanfordNLP.operator_cost = self.process_time / self.process_count

    def _process(self, one: Record, model=None) -> Any:
        k = super()._process(one, model=model)
        if k is None:
            return ''
        else:
            return [e["ner"] for s in k['sentences'] for e in s["entitymentions"]]

    def get_new_object(self, previous_operators=None, operator_name: str = ""):
        """
        return a new object
        :param previous_operators: List[Operator]
        :param operator_name: str, default value is ""
        """
        return EntityStanfordNLP()


"""
test units
"""


def operator_name_test():
    operator = EntityStanfordNLP()
    model = operator.load_model()
    print(operator)
    model.close()


def operator_test():
    """
    operator block process test
    """
    scan = TextJsonScan(TEXT_TWITTER_TRAIN_DATA_PATH)
    block_records = scan.get_block_records(10)
    operator = EntityStanfordNLP()
    model = operator.load_model()
    batch_output = operator.process_batch(batch=block_records, model=model)
    for record in batch_output.returned_batch:
        print(str(operator.get_result(record)).encode('utf-8'))
    print(batch_output)
    scan.close()
    model.close()


def operator_copy_test():
    time0 = time.time()
    operator = EntityStanfordNLP()
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
    block_records = scan.get_block_records(200)
    operator = EntityStanfordNLP()
    model = operator.load_model()
    batch_output = operator.process_batch(batch=block_records, model=model)
    operator.get_operator_cost()
    print("operator cost = " + str(operator.operator_cost) + "\t " + str(batch_output) + "\n")
    time2 = time.time()

    block_records = scan.get_block_records(200)
    batch_output = operator.process_batch(batch=block_records, model=model)
    operator.get_operator_cost()
    print("operator cost = " + str(operator.operator_cost) + "\t " + str(batch_output) + "\n")

    operator_new = EntityStanfordNLP()
    print("new operator cost = " + str(operator_new.operator_cost) + "\n")

    scan.close()
    model.close()
    time3 = time.time()
    print("one process: one block time = " + str(time2 - time1) + "\t two blocks time = " + str(time3 - time2) + "\n\n")


def multple_threads_test():
    time1 = time.time()
    scan = TextJsonScan(TEXT_TWITTER_TRAIN_DATA_PATH)
    block_records = scan.get_block_records(200)
    operator = EntityStanfordNLP()
    model = operator.load_model()
    batch_output = operator.multiple_threads_batch(batch=block_records, num_process=60, model=model)
    operator.get_operator_cost()
    print("operator cost = " + str(operator.operator_cost) + "\t " + str(batch_output) + "\n")
    time2 = time.time()

    block_records = scan.get_block_records(200)
    batch_output = operator.multiple_threads_batch(batch=block_records, num_process=60, model=model)
    operator.get_operator_cost()
    print("operator cost = " + str(operator.operator_cost) + "\t " + str(batch_output) + "\n")

    operator_new = EntityStanfordNLP()
    print("new operator cost = " + str(operator_new.operator_cost) + "\n")

    scan.close()
    model.close()
    time3 = time.time()
    print("one process: one block time = " + str(time2 - time1) + "\t two blocks time = " + str(time3 - time2) + "\n\n")


def multple_processes_test():
    time1 = time.time()
    scan = TextJsonScan(TEXT_TWITTER_TRAIN_DATA_PATH)
    block_records = scan.get_block_records(200)
    operator = EntityStanfordNLP()
    batch_output = operator.multiple_processes_batch(batch=block_records, num_process=10)
    operator.get_operator_cost()
    print("operator cost = " + str(operator.operator_cost) + "\t " + str(batch_output) + "\n")
    time2 = time.time()

    block_records = scan.get_block_records(200)
    batch_output = operator.multiple_processes_batch(batch=block_records, num_process=10)
    operator.get_operator_cost()
    print("operator cost = " + str(operator.operator_cost) + "\t " + str(batch_output) + "\n")

    operator_new = EntityStanfordNLP()
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
    # multple_processes_test()
