import pickle
import time
from typing import Any
from sklearn.feature_extraction.text import CountVectorizer
import rootpath

rootpath.append()
from optimizer.PPs.preprocessor.preprocessor import Preprocessor
from operators.scan.text_json_scan import TextJsonScan
from records.record import Record
from paths import VOCAB_PATH, TEXT_TWITTER_TRAIN_DATA_PATH


class TextPreprocessor(Preprocessor):
    """
    this operator convert text into features.
    preprocess text: convert text into class 'scipy.sparse.csr.csr_matrix'
    """

    def __init__(self):
        super(TextPreprocessor, self).__init__()

    def get_operator_cost(self):
        """return operator time cost per record, in ms/record
            :raises ValueError
        """
        if TextPreprocessor.operator_cost is None:
            if self.process_count == 0:
                print("Getting operator cost: process_count = " + str(
                    self.process_count) + "\tYou may forget add processing statistic informations ...")
                TextPreprocessor.operator_cost = None
            else:
                TextPreprocessor.operator_cost = self.process_time / self.process_count

    def load_model(self) -> Any:
        time1 = time.time()
        vocab = pickle.load(open(VOCAB_PATH, 'rb'))
        vectorizer = CountVectorizer(vocabulary=vocab)
        print(self.operator_name + ": finish loading model. Loading model cost = " + str(time.time() - time1))
        return vectorizer

    def _process(self, one: Record, model=None) -> Any:
        result = model.fit_transform([one.value])
        return result

    def get_new_object(self, previous_operators=None, operator_name: str = ""):
        """
        return a new object
        :param previous_operators: List[Operator]
        :param operator_name: str, default value is ""
        """
        return TextPreprocessor()


"""
test units
"""


def operator_name_test():
    text_preprocessor = TextPreprocessor()
    text_preprocessor.load_model()
    print(text_preprocessor)


def operator_test():
    """
    text_preprocessor block process test
    :return:
    """
    scan = TextJsonScan(TEXT_TWITTER_TRAIN_DATA_PATH)
    block_records = scan.get_block_records(1000)
    operator = TextPreprocessor()
    model = operator.load_model()
    batch_output = operator.process_batch(batch=block_records, model=model)
    for record in batch_output.returned_batch:
        print(str(operator.get_result(record)).encode('utf-8'))
    operator.get_operator_cost()
    operator.print_operator_detail()
    print(batch_output)
    print(batch_output.returned_batch[0])
    scan.close()


def operator_copy_test():
    time0 = time.time()
    operator = TextPreprocessor()
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
    operator = TextPreprocessor()
    model = operator.load_model()
    batch_output = operator.process_batch(batch=block_records, model=model)
    operator.get_operator_cost()
    operator.print_operator_detail()
    print(batch_output)
    print(batch_output.returned_batch[0])
    print("\n")
    time2 = time.time()

    block_records = scan.get_block_records(200)
    batch_output = operator.process_batch(batch=block_records, model=model)
    operator.get_operator_cost()
    operator.print_operator_detail()
    print(batch_output)
    print(batch_output.returned_batch[0])
    print("\n")

    operator_new = TextPreprocessor()
    print("new operator cost = " + str(operator_new.operator_cost) + "\n")

    scan.close()
    time3 = time.time()
    print("one process: one block time = " + str(time2 - time1) + "\t two blocks time = " + str(time3 - time2) + "\n\n")


def multple_threads_test():
    time1 = time.time()
    scan = TextJsonScan(TEXT_TWITTER_TRAIN_DATA_PATH)
    block_records = scan.get_block_records(200)
    operator = TextPreprocessor()
    model = operator.load_model()
    batch_output = operator.multiple_threads_batch(batch=block_records, num_process=10, model=model)
    operator.get_operator_cost()
    operator.print_operator_detail()
    print(batch_output)
    print(batch_output.returned_batch[0])
    print("\n")
    time2 = time.time()

    block_records = scan.get_block_records(200)
    batch_output = operator.multiple_threads_batch(batch=block_records, num_process=10, model=model)
    operator.get_operator_cost()
    operator.print_operator_detail()
    print(batch_output)
    print(batch_output.returned_batch[0])
    print("\n")

    operator_new = TextPreprocessor()
    print("new operator cost = " + str(operator_new.operator_cost) + "\n")

    scan.close()
    time3 = time.time()
    print("multiple process: one block time = " + str(time2 - time1) + "\t two blocks time = " + str(
        time3 - time2) + "\n")


def multple_processes_test():
    time1 = time.time()
    scan = TextJsonScan(TEXT_TWITTER_TRAIN_DATA_PATH)
    block_records = scan.get_block_records(200)
    operator = TextPreprocessor()
    batch_output = operator.multiple_processes_batch(batch=block_records, num_process=10)
    operator.get_operator_cost()
    operator.print_operator_detail()
    print(batch_output)
    print(batch_output.returned_batch[0])
    print("\n")
    time2 = time.time()

    block_records = scan.get_block_records(200)
    batch_output = operator.multiple_processes_batch(batch=block_records, num_process=10)
    operator.get_operator_cost()
    operator.print_operator_detail()
    print(batch_output)
    print(batch_output.returned_batch[0])
    print("\n")

    operator_new = TextPreprocessor()
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
