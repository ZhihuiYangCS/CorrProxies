from typing import List, Union

import rootpath

rootpath.append()


class AccuracyInfoDict:
    """
    an abstract of information of accuracy, reduction, threshold, et.al.
    """

    def __init__(self, index: int, train_cost: float, infer_cost: float, target_accuracy: float, final_accuracy: float,
                 threshold: float, reduction: float):
        self.index = index
        self.train_cost = train_cost
        self.infer_cost = infer_cost
        self.target_accuracy = target_accuracy
        self.final_accuracy = final_accuracy
        self.threshold = threshold
        self.reduction = reduction

    def __str__(self):
        return f"[index = {self.index}, train_cost = {self.train_cost}, infer_cost = " \
               f"{self.infer_cost}, target_accuracy = {self.target_accuracy}, final_accuracy = {self.final_accuracy}," \
               f"threshold = {self.threshold}, reduction = {self.reduction}]"

    __repr__ = __str__


class AccuracyInfos:
    """
    an abstract of a PPs accuracy_reduction mapping relationship. The relationship is a list of AccuracyInfoDict
    """

    def __init__(self, accuracy_info_list: List[AccuracyInfoDict]):
        self.accuracy_info_list = accuracy_info_list

    def get_accuracy_info(self, accuracy: float) -> AccuracyInfoDict:
        """
        get accuracy_info for a specified accuracy
        :param accuracy:
        :return: an AccuracyInfoDict
        """
        start_index = 0
        end_index = len(self.accuracy_info_list) - 2
        result = self._is_equal(start_index, accuracy)
        if result is not None:
            return result
        else:
            result = self._is_equal(end_index, accuracy)
            if result is not None:
                return result
            else:
                while start_index <= end_index:
                    mid_index = int((start_index + end_index) / 2)
                    result = self._is_equal(mid_index, accuracy)
                    if result is not None:
                        return result
                    elif accuracy < self.accuracy_info_list[mid_index].target_accuracy:
                        start_index = mid_index + 1
                    else:
                        end_index = mid_index - 1
        if accuracy < self.accuracy_info_list[-1].target_accuracy:
            return self.accuracy_info_list[-1]
        if accuracy > 1.0:
            return self.accuracy_info_list[0]

    def _is_equal(self, index: int, accuracy: float) -> Union[AccuracyInfoDict, None]:
        if accuracy == self.accuracy_info_list[index].target_accuracy:
            return self.accuracy_info_list[index]
        elif self.accuracy_info_list[index].target_accuracy > accuracy >= self.accuracy_info_list[
            index + 1].target_accuracy:
            return self.accuracy_info_list[index + 1]
        else:
            return None

    def __str__(self):
        info = '\n '.join(map(str, self.accuracy_info_list))
        return f"accuracy_infos:\n[{info}]"

    __repr__ = __str__
