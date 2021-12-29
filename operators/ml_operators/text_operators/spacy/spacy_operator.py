import time
from abc import ABC
from typing import Any

import rootpath


rootpath.append()
from operators.scan.text_json_scan import TextJsonScan
from paths import TEXT_TWITTER_TRAIN_DATA_PATH
from records.record import Record
from operators.operator_base.operator_parallel import OperatorParallel

try:
    import en_core_web_lg
    import en_core_web_sm
except:
    import subprocess
    import sys


    def download(corpus):
        subprocess.check_call([sys.executable, "-m", "spacy", "download", corpus])


    download('en_core_web_lg')
    download('en_core_web_sm')
    import en_core_web_lg
    import en_core_web_sm


class SpacyOperator(OperatorParallel, ABC):
    """
    base class to process one record
    """

    def __init__(self):
        super().__init__()

    def _process(self, one: Record, model=None) -> Any:
        doc = model(one.value)
        result = dict()
        for token in doc:
            result[str(token)] = [token.tag_, token.pos_, token.shape_]
        return result

    def get_new_object(self, previous_operators=None, operator_name: str = ""):
        """
        return a new object
        :param previous_operators: List[Operator]
        :param operator_name: str, default value is ""
        """
        if isinstance(self, POSTaggerSpacySM):
            return POSTaggerSpacySM()
        elif isinstance(self, POSTaggerSpacyLG):
            return POSTaggerSpacyLG()
        else:
            return None


class POSTaggerSpacySM(SpacyOperator):
    """
    define and initialize sel.nlp
    """

    def __init__(self):
        super(POSTaggerSpacySM, self).__init__()

    def get_operator_cost(self):
        """return operator time cost per record, in ms/record
            :raises ValueError
        """
        if POSTaggerSpacySM.operator_cost is None:
            if self.process_count == 0:
                print("Getting operator cost: process_count = " + str(
                    self.process_count) + "\tYou may forget add processing statistic informations ...")
                POSTaggerSpacySM.operator_cost = None
            else:
                POSTaggerSpacySM.operator_cost = self.process_time / self.process_count

    def load_model(self) -> Any:
        time1 = time.time()
        model = en_core_web_sm.load()
        print(self.operator_name + ": finish loading model. Loading model cost = " + str(time.time() - time1))
        return model


class POSTaggerSpacyLG(SpacyOperator):
    """
        define and initialize sel.nlp
        """

    def __init__(self):
        super(POSTaggerSpacyLG, self).__init__()

    def get_operator_cost(self):
        """return operator time cost per record, in ms/record
            :raises ValueError
        """
        if POSTaggerSpacyLG.operator_cost is None:
            if self.process_count == 0:
                print("Getting operator cost: process_count = " + str(
                    self.process_count) + "\tYou may forget add processing statistic informations ...")
                POSTaggerSpacyLG.operator_cost = None
            else:
                POSTaggerSpacyLG.operator_cost = self.process_time / self.process_count

    def load_model(self) -> Any:
        time1 = time.time()
        model = en_core_web_lg.load()
        print(self.operator_name + ": finish loading model. Loading model cost = " + str(time.time() - time1))
        return model


"""
test units
"""


def operator_name_test():
    """
    pos_tagger_spacy operators test
    """
    pos_sm = POSTaggerSpacySM()
    model_sm = pos_sm.load_model()
    pos_lg = POSTaggerSpacyLG()
    model_lg = pos_lg.load_model()
    print(pos_sm)
    print(pos_lg)


def operator_test():
    """
    pos_tagger_spacy operators block process test
    """
    operators, models = [], []
    pos_tagger_spacy_sm = POSTaggerSpacySM()
    operators.append(pos_tagger_spacy_sm)
    models.append(pos_tagger_spacy_sm.load_model())

    pos_tagger_spacy_lg = POSTaggerSpacyLG()
    operators.append(pos_tagger_spacy_lg)
    models.append(pos_tagger_spacy_lg.load_model())

    for index, operator in enumerate(operators):
        print("start test " + operator.operator_name + " operator")
        scan = TextJsonScan(TEXT_TWITTER_TRAIN_DATA_PATH)
        block_records = scan.get_block_records(10)
        batch_output = operator.process_batch(batch=block_records, model=models[index])
        for record in batch_output.returned_batch:
            print(str(operator.get_result(record)).encode('utf-8'))
        print(operator.operator_name + "\t" + str(batch_output))
        scan.close()


def operator_copy_test():
    create_time = []
    copy_time = []
    operators = []
    time0 = time.time()
    pos_tagger_spacy_sm = POSTaggerSpacySM()
    create_time.append(time.time() - time0)
    operators.append(pos_tagger_spacy_sm)

    time1 = time.time()
    pos_tagger_spacy_lg = POSTaggerSpacyLG()
    create_time.append(time.time() - time1)
    operators.append(pos_tagger_spacy_lg)

    for operator in operators:
        print(operator.operator_name)
        time4 = time.time()
        operator_copy = operator.get_new_object()
        copy_time.append(time.time() - time4)

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
    print("create time = " + str(create_time))
    print("copy time = " + str(copy_time))


def operator_cost_test():
    """
    test calculate operator cost of the Operator base class.
    :return:
    """
    operators, models = [], []
    pos_tagger_spacy_sm = POSTaggerSpacySM()
    operators.append(pos_tagger_spacy_sm)
    models.append(pos_tagger_spacy_sm.load_model())

    pos_tagger_spacy_lg = POSTaggerSpacyLG()
    operators.append(pos_tagger_spacy_lg)
    models.append(pos_tagger_spacy_lg.load_model())

    for index, operator in enumerate(operators):
        time1 = time.time()
        print("start test " + operator.operator_name + " operator")
        scan = TextJsonScan(TEXT_TWITTER_TRAIN_DATA_PATH)
        block_records = scan.get_block_records(200)
        batch_output = operator.process_batch(batch=block_records, model=models[index])
        operator.get_operator_cost()
        print("operator cost = " + str(operator.operator_cost) + "\t " + str(batch_output) + "\n")
        time2 = time.time()

        block_records = scan.get_block_records(200)
        batch_output = operator.process_batch(batch=block_records, model=models[index])
        operator.get_operator_cost()
        print("operator cost = " + str(operator.operator_cost) + "\t " + str(batch_output) + "\n")

        scan.close()
        time3 = time.time()
        print("one process: one block time = " + str(time2 - time1) + "\t two blocks time = " + str(
            time3 - time2) + "\n\n")


def multple_threads_test():
    """
    test calculate operator cost of the Operator base class.
    :return:
    """
    operators, models = [], []
    pos_tagger_spacy_sm = POSTaggerSpacySM()
    operators.append(pos_tagger_spacy_sm)
    models.append(pos_tagger_spacy_sm.load_model())

    pos_tagger_spacy_lg = POSTaggerSpacyLG()
    operators.append(pos_tagger_spacy_lg)
    models.append(pos_tagger_spacy_lg.load_model())

    for index, operator in enumerate(operators):
        time1 = time.time()
        print("start test " + operator.operator_name + " operator")
        scan = TextJsonScan(TEXT_TWITTER_TRAIN_DATA_PATH)
        block_records = scan.get_block_records(200)
        batch_output = operator.multiple_threads_batch(batch=block_records, num_process=10, model=models[index])
        operator.get_operator_cost()
        print("operator cost = " + str(operator.operator_cost) + "\t " + str(batch_output) + "\n")
        time2 = time.time()

        block_records = scan.get_block_records(200)
        batch_output = operator.multiple_threads_batch(batch=block_records, num_process=10, model=models[index])
        operator.get_operator_cost()
        print("operator cost = " + str(operator.operator_cost) + "\t " + str(batch_output) + "\n")

        scan.close()
        time3 = time.time()
        print("one process: one block time = " + str(time2 - time1) + "\t two blocks time = " + str(
            time3 - time2) + "\n\n")


def multple_processes_test():
    """
    test calculate operator cost of the Operator base class.
    :return:
    """
    operators = []
    pos_tagger_spacy_sm = POSTaggerSpacySM()
    operators.append(pos_tagger_spacy_sm)

    pos_tagger_spacy_lg = POSTaggerSpacyLG()
    operators.append(pos_tagger_spacy_lg)

    for index, operator in enumerate(operators):
        time1 = time.time()
        print("start test " + operator.operator_name + " operator")
        scan = TextJsonScan(TEXT_TWITTER_TRAIN_DATA_PATH)
        block_records = scan.get_block_records(200)
        batch_output = operator.multiple_processes_batch(batch=block_records, num_process=10)
        operator.get_operator_cost()
        print("operator cost = " + str(operator.operator_cost) + "\t " + str(batch_output) + "\n")
        time2 = time.time()

        block_records = scan.get_block_records(200)
        batch_output = operator.multiple_processes_batch(batch=block_records, num_process=10)
        operator.get_operator_cost()
        print("operator cost = " + str(operator.operator_cost) + "\t " + str(batch_output) + "\n")

        scan.close()
        time3 = time.time()
        print("one process: one block time = " + str(time2 - time1) + "\t two blocks time = " + str(
            time3 - time2) + "\n\n")


if __name__ == '__main__':
    pass
#     operator_name_test()
#     operator_test()
#     operator_copy_test()
#     operator_cost_test()
#     multple_threads_test()
    # multple_processes_test()
