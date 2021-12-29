# from copy import deepcopy
import copy
import json
from typing import Any, Union


class Record:
    def __init__(self, id: int, value: Union[str]):
        self.id = id
        self.value = value

    @classmethod
    def json2record(cls, json_data):
        id_value = 0
        value = ""
        json_data = json.loads(json_data)
        if "id" in json_data:
            id_value = int(json_data["id"])
        if "value" in json_data:
            value = json_data["value"]
        record = cls(id=id_value, value=value)
        for k, v in json_data.items():
            record.__setitem__(key=k, value=v)
        return record

    def __setitem__(self, key, value):
        self.__dict__[key] = value

    def __getitem__(self, item) -> Union[Any, None]:
        return self.__dict__.get(item)

    def __delitem__(self, key):
        self.__dict__.pop(key, None)

    def __str__(self):
        return f"Record[{', '.join(map(lambda x: f'{x[0]}={x[1]}', self.__dict__.items()))}]"

    def __copy__(self):
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update(self.__dict__)
        return result

    def __deepcopy__(self, memodict={}):
        cls = self.__class__
        result = cls.__new__(cls)
        memodict[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, copy.deepcopy(v, memodict))
        return result

    def del_key_value(self, key: str):
        """
        delete key_value item in the record
        :param key: a key to be deleted
        :return: the record itself with key_value item deleted
        """
        self.__delitem__(key=key)

    def copy(self):
        """
        copy a record
        :return: a copy of the record
        """
        return self.__copy__()

    __repr__ = __str__

    def get_value(self, key: str):
        """
        return key's value
        :param key: key name in record
        :return: value for the key
        """
        return self.__dict__.get(key)

    def get_keys(self):
        """
        return all keys in the record
        :return: all keys
        """
        return self.__dict__.keys()

    def record2json(self):
        dict_data = {k: v for k, v in self.__dict__.items()}
        json_data = json.dumps(dict_data)
        return json_data


"""
test units
"""


def record_add_test():
    print("record add test:")
    record = Record(id=1, value="hello")
    print("\t" + str(record))
    record["key"] = "value"
    print("\t" + str(record))


def record_get_test():
    print("record_get_test")
    record = Record(id=1, value="hello")
    record["key"] = "value"
    print("\t" + str(record.get_keys()))
    print("\trecord.key = " + str(record.get_value("key")))


def copy_test():
    print("copy test")
    record = Record(id=1, value="hello")
    record_copy = record.copy()
    print("\tbefore change:")
    print("\t\trecord = " + str(record))
    print("\t\trecord copy = " + str(record_copy))
    record.id = 2
    print("\tafter change record:")
    print("\t\trecord = " + str(record))
    print("\t\trecord copy = " + str(record_copy))
    record_copy.id = 3
    print("\tafter change record:")
    print("\t\trecord = " + str(record))
    print("\t\trecord copy = " + str(record_copy))


def deepcopy_test():
    print("deepcopy test")
    record = Record(id=1, value="hello")
    record_deepcopy = record.__deepcopy__()
    print("\tbefore change:")
    print("\t\trecord = " + str(record))
    print("\t\trecord copy = " + str(record_deepcopy))
    record.id = 2
    print("\tafter change record:")
    print("\t\trecord = " + str(record))
    print("\t\trecord copy = " + str(record_deepcopy))
    record_deepcopy.id = 3
    print("\tafter change record:")
    print("\t\trecord = " + str(record))
    print("\t\trecord copy = " + str(record_deepcopy))


def record_list_copy_test():
    print("record_list_test")
    record_list = []
    record1 = Record(id=1, value="hello")
    record2 = Record(id=2, value="world")
    record_list.append(record1)
    record_list.append(record2)
    record_list_copy = record_list.copy()
    print("\tbefore change:")
    print("\t\trecord_list" + str(record_list))
    print("\t\trecord_list_copy = " + str(record_list_copy))
    record_list[0].id = 3
    record_list[0].value = "record3"
    print("\tafter change:")
    print("\t\trecord_list" + str(record_list))
    print("\t\trecord_list_copy = " + str(record_list_copy))


def record_list_deepcopy_test():
    print("record_list_deepcopy_test")
    record_list = []
    record1 = Record(id=1, value="hello")
    record2 = Record(id=2, value="world")
    record_list.append(record1)
    record_list.append(record2)
    record_list_copy = copy.deepcopy(record_list)
    print("\tbefore change:")
    print("\t\trecord_list" + str(record_list))
    print("\t\trecord_list_copy = " + str(record_list_copy))
    record_list[0].id = 3
    record_list[0].value = "record3"
    print("\tafter change:")
    print("\t\trecord_list" + str(record_list))
    print("\t\trecord_list_copy = " + str(record_list_copy))


def delete_test():
    print("record add test:")
    record = Record(id=1, value="hello")
    print("\t" + str(record))
    record["key"] = "value"
    print("\t" + str(record))
    record.del_key_value('key')
    print("\t" + str(record))
    record.del_key_value('key')
    print("\t" + str(record))

def convert2json_test():
    record = Record(id=1, value="hello")
    json_data = record.record2json()
    print(json_data)
    record = Record.json2record(json_data=json_data)
    print(record)


if __name__ == '__main__':
    # record_add_test()
    # record_get_test()
    # copy_test()
    # deepcopy_test()
    # record_list_copy_test()
    # record_list_deepcopy_test()
    # delete_test()
    convert2json_test()
