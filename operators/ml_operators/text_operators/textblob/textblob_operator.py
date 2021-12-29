import time
from abc import ABC
from typing import Any

import rootpath


rootpath.append()
from textblob import TextBlob
from operators.scan.text_json_scan import TextJsonScan
from paths import TEXT_TWITTER_TRAIN_DATA_PATH
from records.record import Record
from operators.operator_base.operator_parallel import OperatorParallel


class TextBlobOperator(OperatorParallel, ABC):
    def __init__(self):
        super(TextBlobOperator, self).__init__()

    def load_model(self) -> Any:
        time1 = time.time()
        model = TextBlob
        print(self.operator_name + ": finish loading model. Loading model cost = " + str(time.time() - time1))
        return model

    def get_new_object(self, previous_operators=None, operator_name: str = ""):
        """
        return a new object
        :param previous_operators: List[Operator]
        :param operator_name: str, default value is ""
        """
        if isinstance(self, POSTaggerTextBlob):
            return POSTaggerTextBlob()
        elif isinstance(self, NounPhaseExtractionTextBlob):
            return NounPhaseExtractionTextBlob()
        # elif isinstance(self, SentimentTextBlob):
        #     return SentimentTextBlob()
        else:
            return None


class POSTaggerTextBlob(TextBlobOperator):
    def __init__(self):
        super(POSTaggerTextBlob, self).__init__()

    def get_operator_cost(self):
        """return operator time cost per record, in ms/record
            :raises ValueError
        """
        if POSTaggerTextBlob.operator_cost is None:
            if self.process_count == 0:
                print("Getting operator cost: process_count = " + str(
                    self.process_count) + "\tYou may forget add processing statistic informations ...")
                POSTaggerTextBlob.operator_cost = None
            else:
                POSTaggerTextBlob.operator_cost = self.process_time / self.process_count

    def _process(self, one: Record, model=None) -> Any:
        return model(one.value).tags


class NounPhaseExtractionTextBlob(TextBlobOperator):
    def __init__(self):
        super(NounPhaseExtractionTextBlob, self).__init__()

    def get_operator_cost(self):
        """return operator time cost per record, in ms/record
            :raises ValueError
        """
        if NounPhaseExtractionTextBlob.operator_cost is None:
            if self.process_count == 0:
                print("Getting operator cost: process_count = " + str(
                    self.process_count) + "\tYou may forget add processing statistic informations ...")
                NounPhaseExtractionTextBlob.operator_cost = None
            else:
                NounPhaseExtractionTextBlob.operator_cost = self.process_time / self.process_count

    def _process(self, one: Record, model=None) -> Any:
        return model(one.value).noun_phrases


# class SentimentTextBlob(TextBlobOperator):
#
#     def _process(self, one: Record):
#         return TextBlob(one.value).sentiment


"""
test units
"""


def operator_name_test():
    pos_tagger = POSTaggerTextBlob()
    pos_tagger_model = pos_tagger.load_model()
    print(pos_tagger)

    noun_phase = NounPhaseExtractionTextBlob()
    noun_phase_model = noun_phase.load_model()
    print(noun_phase)

    # sentiment = SentimentTextBlob()
    # sentiment_model = sentiment.load_model()
    # print(sentiment)


def operator_test():
    """
    textblob operators block process test
    """
    operators, models = [], []
    pos_tagger = POSTaggerTextBlob()
    operators.append(pos_tagger)
    models.append(pos_tagger.load_model())

    noun_phase = NounPhaseExtractionTextBlob()
    operators.append(noun_phase)
    models.append(noun_phase.load_model())

    # sentiment = SentimentTextBlob()
    # operators.append(sentiment)
    # models.append(sentiment.load_model())

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
    pos_tagger = POSTaggerTextBlob()
    create_time.append(time.time() - time0)
    operators.append(pos_tagger)

    time1 = time.time()
    noun_phase = NounPhaseExtractionTextBlob()
    create_time.append(time.time() - time1)
    operators.append(noun_phase)

    # time2 = time.time()
    # sentiment = SentimentTextBlob()
    # create_time.append(time.time() - time2)
    # operators.append(sentiment)

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

def calculate_operator_cost_test():
    """
    test calculate operator cost of the Operator base class.
    """
    operators = []
    pos_tagger = POSTaggerTextBlob()
    operators.append(pos_tagger)

    noun_phase = NounPhaseExtractionTextBlob()
    operators.append(noun_phase)

    # sentiment = SentimentTextBlob()
    # operators.append(sentiment)

    for operator in operators:
        print("start test " + operator.operator_name + " operator")
        scan = TextJsonScan(TEXT_TWITTER_TRAIN_DATA_PATH)
        block_records = scan.get_block_records(1000)
        operator.process_batch(block_records)
        print(operator.operator_name + " cost = " + str(operator.get_operator_cost()))
        scan.close()


def operator_cost_test():
    """
    test calculate operator cost of the Operator base class.
    :return:
    """
    operators, models = [], []
    pos_tagger = POSTaggerTextBlob()
    operators.append(pos_tagger)
    models.append(pos_tagger.load_model())

    noun_phase = NounPhaseExtractionTextBlob()
    operators.append(noun_phase)
    models.append(noun_phase.load_model())

    # sentiment = SentimentTextBlob()
    # operators.append(sentiment)
    # models.append(sentiment.load_model())

    for index, operator in enumerate(operators):
        time1 = time.time()
        print("start test " + operator.operator_name + " operator")
        scan = TextJsonScan(TEXT_TWITTER_TRAIN_DATA_PATH)
        block_records = scan.get_block_records(100)
        batch_output = operator.process_batch(batch=block_records, model=models[index])
        operator.get_operator_cost()
        print("operator cost = " + str(operator.operator_cost) + "\t " + str(batch_output) + "\n")
        time2 = time.time()

        block_records = scan.get_block_records(100)
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
    pos_tagger = POSTaggerTextBlob()
    operators.append(pos_tagger)
    models.append(pos_tagger.load_model())

    noun_phase = NounPhaseExtractionTextBlob()
    operators.append(noun_phase)
    models.append(noun_phase.load_model())

    # sentiment = SentimentTextBlob()
    # operators.append(sentiment)
    # models.append(sentiment.load_model())

    for index, operator in enumerate(operators):
        time1 = time.time()
        print("start test " + operator.operator_name + " operator")
        scan = TextJsonScan(TEXT_TWITTER_TRAIN_DATA_PATH)
        block_records = scan.get_block_records(100)
        batch_output = operator.multiple_threads_batch(batch=block_records, num_process=10, model=models[index])
        operator.get_operator_cost()
        print("operator cost = " + str(operator.operator_cost) + "\t " + str(batch_output) + "\n")
        time2 = time.time()

        block_records = scan.get_block_records(100)
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
    pos_tagger = POSTaggerTextBlob()
    operators.append(pos_tagger)

    noun_phase = NounPhaseExtractionTextBlob()
    operators.append(noun_phase)

    # sentiment = SentimentTextBlob()
    # operators.append(sentiment)

    for index, operator in enumerate(operators):
        time1 = time.time()
        print("start test " + operator.operator_name + " operator")
        scan = TextJsonScan(TEXT_TWITTER_TRAIN_DATA_PATH)
        block_records = scan.get_block_records(100)
        batch_output = operator.multiple_processes_batch(batch=block_records, num_process=10)
        operator.get_operator_cost()
        print("operator cost = " + str(operator.operator_cost) + "\t " + str(batch_output) + "\n")
        time2 = time.time()

        block_records = scan.get_block_records(100)
        batch_output = operator.multiple_processes_batch(batch=block_records, num_process=10)
        operator.get_operator_cost()
        print("operator cost = " + str(operator.operator_cost) + "\t " + str(batch_output) + "\n")

        scan.close()
        time3 = time.time()
        print("one process: one block time = " + str(time2 - time1) + "\t two blocks time = " + str(
            time3 - time2) + "\n\n")


if __name__ == '__main__':
#     operator_name_test()
#     operator_test()
#     operator_copy_test()
#     operator_cost_test()
#     multple_threads_test()
#     multple_processes_test()
