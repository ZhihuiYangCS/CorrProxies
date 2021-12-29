import json
import time
from typing import Any

import rootpath


rootpath.append()
from operators.operator_base.operator_parallel import OperatorParallel
from stanfordcorenlp import StanfordCoreNLP
from paths import STANFORD_NLP_MODEL_PATH
from records.record import Record


class StanfordNLPOperator(OperatorParallel):
    """
    a base operator for stanford nlp package inheriting from OperatorDynamicAction
    """

    def __init__(self):
        super(StanfordNLPOperator, self).__init__()
        self.properties = None

    def load_model(self) -> Any:
        time1 = time.time()
        model = StanfordCoreNLP(STANFORD_NLP_MODEL_PATH)
        print(self.operator_name + ": finish loading model. Loading model cost = " + str(time.time() - time1))
        return model

    def _process(self, one: Record, model=None) -> Any:
        result = model.annotate(one.value, properties=self.properties)
        json_result = None
        try:
            json_result = json.loads(result)
        except:
            print("skip tweet\t" + str(one))
            # exit(1)
        return json_result
        # return json.loads(result)
