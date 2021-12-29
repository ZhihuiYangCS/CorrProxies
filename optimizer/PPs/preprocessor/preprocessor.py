from abc import ABC

import rootpath


rootpath.append()
from operators.operator_base.operator_parallel import OperatorParallel


class Preprocessor(OperatorParallel, ABC):
    def __init__(self):
        super(Preprocessor, self).__init__()
