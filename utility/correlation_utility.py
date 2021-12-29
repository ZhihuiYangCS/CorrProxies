import json
import operator
from typing import List
import rootpath


rootpath.append()
from records.record import Record
from operators.ml_operators.image_video_operators.image_object_detection_yolov3 import ImageObjectDetectionYolov3
from operators.ml_operators.image_video_operators.video_activity_recognition import VideoActivityRecognition
from operators.ml_operators.image_video_operators.video_object_detection_yolov3 import VideoObjectDetectionYolov3
from operators.ml_operators.text_operators.spacy.spacy_operator import POSTaggerSpacyLG, POSTaggerSpacySM
from operators.ml_operators.text_operators.stanfordnlp.entity_stanfordnlp import EntityStanfordNLP
from operators.ml_operators.text_operators.stanfordnlp.pos_tagger_stanfordnlp import POSTaggerStanfordNLP
from operators.ml_operators.text_operators.stanfordnlp.sentiment_stanfordnlp import SentimentStanfordNLP
from operators.operator_base.operator_parallel import OperatorParallel



def read_records(path: str):
    block_records = []
    with open(path, 'r') as reader:
        for line in reader:
            json_data = json.loads(line)
            record = Record.json2record(json_data=json_data)
            block_records.append(record)
    return block_records


def get_pos_tagger_stanfordnlp_categories(ml_operator: OperatorParallel, block_records: List[Record],
                                          one_category: list) -> list:
    pos_tagger_value_recordIDs = {}
    pos_tagger_value_count = {}
    pos_tagger_categories = []
    for record in block_records:
        pos_record_result = ml_operator.get_result(record)
        pos_record_set = set()
        for pos in pos_record_result:
            pos_record_set.add(pos)
        for pos in pos_record_set:
            if pos in pos_tagger_value_recordIDs.keys():
                pos_tagger_value_recordIDs[pos].append(record.id)
                pos_tagger_value_count[pos] += 1
            else:
                pos_tagger_value_recordIDs[pos] = [record.id]
                pos_tagger_value_count[pos] = 1
    frequent_records = set()
    pos_tagger_categories.append(one_category)
    for item in one_category:
        if item in pos_tagger_value_count.keys():
            frequent_records = frequent_records.union(set(pos_tagger_value_recordIDs[item]))
            del pos_tagger_value_count[item]
    sorted_pos_tagger_value_count = dict(
        sorted(pos_tagger_value_count.items(), key=operator.itemgetter(1), reverse=True))

    for key, value in sorted_pos_tagger_value_count.items():
        if len(frequent_records) < 0.95 * len(block_records):
            pos_tagger_categories.append([key])
            frequent_records = frequent_records.union(set(pos_tagger_value_recordIDs[key]))
        else:
            break
    return pos_tagger_categories


def get_pos_tagger_stanfordnlp_category_label(ml_operator: OperatorParallel, record: Record, category: list) -> bool:
    category_set = set(category)
    result = ml_operator.get_result(record)
    return any([pos in category_set for pos in result])


def get_sentiment_stanfordnlp_categories(ml_operator: OperatorParallel, block_records: List[Record],
                                         one_category: list) -> list:
    sentiment_categories = [one_category]
    category = [1, 2, 3]
    difference = list(list(set(category) - set(one_category)) + list(set(one_category) - set(category)))
    for item in difference:
        sentiment_categories.append([item])
    return sentiment_categories


def get_sentiment_stanfordnlp_category_label(ml_operator: OperatorParallel, record: Record, category: list) -> bool:
    sentiment_record_result = ml_operator.get_result(record)
    long_sentence = 0
    sentiment = 0
    for s in sentiment_record_result:
        sentence = " ".join([t["word"] for t in s["tokens"]])
        if long_sentence < len(sentence):
            long_sentence = len(sentence)
            sentiment = s["sentimentValue"]
    if (int(sentiment) < 2 and 1 in category) or (int(sentiment) == 2 and 2 in category) or (
            int(sentiment) > 2 and 3 in category):
        return True
    else:
        return False


def get_entity_stanfordnlp_categories(ml_operator: OperatorParallel, block_records: List[Record], one_category: list) -> list:
    entity_value_recordIDs = {}
    entity_value_count = {}
    entity_categories = []
    for record in block_records:
        entity_record_result = ml_operator.get_result(record)
        entity_record_set = set()
        for entity in entity_record_result:
            entity_record_set.add(entity)
        for entity in entity_record_set:
            if entity in entity_value_recordIDs.keys():
                entity_value_recordIDs[entity].append(record.id)
                entity_value_count[entity] += 1
            else:
                entity_value_recordIDs[entity] = [record.id]
                entity_value_count[entity] = 1
    frequent_records = set()
    entity_categories.append(one_category)
    for item in one_category:
        if item in entity_value_count.keys():
            frequent_records = frequent_records.union(set(entity_value_recordIDs[item]))
            del entity_value_count[item]
    sorted_entity_value_count = dict(
        sorted(entity_value_count.items(), key=operator.itemgetter(1), reverse=True))

    for key, value in sorted_entity_value_count.items():
        if len(frequent_records) < 0.95 * len(block_records):
            entity_categories.append([key])
            frequent_records = frequent_records.union(set(entity_value_recordIDs[key]))
        else:
            break
    return entity_categories


def get_entity_stanfordnlp_category_label(ml_operator: OperatorParallel, record: Record, category: list) -> bool:
    category_set = set(category)
    result = ml_operator.get_result(record)
    return any([entity in category_set for entity in result])


def get_pos_tagger_spacy_categories(ml_operator: OperatorParallel, block_records: List[Record], one_category: list) -> list:
    pos_tagger_value_recordIDs = {}
    pos_tagger_value_count = {}
    pos_tagger_categories = []
    for record in block_records:
        pos_record_result = ml_operator.get_result(record)
        pos_record_set = set()
        for token, value in pos_record_result.items():
            pos_record_set.add(value[0])
        for pos in pos_record_set:
            if pos in pos_tagger_value_count.keys():
                pos_tagger_value_recordIDs[pos].append(record.id)
                pos_tagger_value_count[pos] += 1
            else:
                pos_tagger_value_recordIDs[pos] = [record.id]
                pos_tagger_value_count[pos] = 1
    frequent_records = set()
    pos_tagger_categories.append(one_category)
    for item in one_category:
        if item in pos_tagger_value_count.keys():
            frequent_records = frequent_records.union(set(pos_tagger_value_recordIDs[item]))
            del pos_tagger_value_count[item]
    sorted_pos_tagger_value_count = dict(
        sorted(pos_tagger_value_count.items(), key=operator.itemgetter(1), reverse=True))

    for key, value in sorted_pos_tagger_value_count.items():
        if len(frequent_records) < 0.95 * len(block_records):
            pos_tagger_categories.append([key])
            frequent_records = frequent_records.union(set(pos_tagger_value_recordIDs[key]))
        else:
            break
    return pos_tagger_categories


def get_pos_tagger_spacy_category_label(ml_operator: OperatorParallel, record: Record, category: list) -> bool:
    category_set = set(category)
    result = ml_operator.get_result(record)
    return any([value[0] in category_set for token, value in result.items()])


def get_image_object_detection_categories(ml_operator: OperatorParallel, block_records: List[Record],
                                          one_category: list) -> list:
    image_od_value_recordIDs = {}
    image_od_value_count = {}
    image_od_categories = []
    for record in block_records:
        image_od_record_result = ml_operator.get_result(record)
        for key in image_od_record_result.keys():
            if key in image_od_value_count.keys():
                image_od_value_recordIDs[key].append(record.id)
                image_od_value_count[key] += 1
            else:
                image_od_value_recordIDs[key] = [record.id]
                image_od_value_count[key] = 1
    frequent_records = set()
    image_od_categories.append(one_category)
    for item in one_category:
        if item in image_od_value_count.keys():
            frequent_records = frequent_records.union(set(image_od_value_recordIDs[item]))
            del image_od_value_count[item]
    sorted_pos_tagger_value_count = dict(
        sorted(image_od_value_count.items(), key=operator.itemgetter(1), reverse=True))

    for key, value in sorted_pos_tagger_value_count.items():
        if len(frequent_records) < 0.95 * len(block_records):
            image_od_categories.append([key])
            frequent_records = frequent_records.union(set(image_od_value_recordIDs[key]))
        else:
            break
    return image_od_categories


def get_image_object_detection_category_label(ml_operator: OperatorParallel, record: Record, category: list) -> bool:
    category_set = set(category)
    result = ml_operator.get_result(record)
    return any([key in category_set for key in result.keys()])


def get_video_object_detection_categories(ml_operator: OperatorParallel, block_records: List[Record],
                                          one_category: list) -> list:
    video_od_value_recordIDs = {}
    video_od_value_count = {}
    video_od_categories = []
    for record in block_records:
        video_od_record_result = ml_operator.get_result(record)
        for key in video_od_record_result.keys():
            if key in video_od_value_count.keys():
                video_od_value_recordIDs[key].append(record.id)
                video_od_value_count[key] += 1
            else:
                video_od_value_recordIDs[key] = [record.id]
                video_od_value_count[key] = 1
    frequent_records = set()
    video_od_categories.append(one_category)
    for item in one_category:
        if item in video_od_value_count.keys():
            frequent_records = frequent_records.union(set(video_od_value_recordIDs[item]))
            del video_od_value_count[item]
    sorted_pos_tagger_value_count = dict(
        sorted(video_od_value_count.items(), key=operator.itemgetter(1), reverse=True))

    for key, value in sorted_pos_tagger_value_count.items():
        if len(frequent_records) < 0.95 * len(block_records):
            video_od_categories.append([key])
            frequent_records = frequent_records.union(set(video_od_value_recordIDs[key]))
        else:
            break
    return video_od_categories


def get_video_object_detection_category_label(ml_operator: OperatorParallel, record: Record, category: list) -> bool:
    category_set = set(category)
    result = ml_operator.get_result(record)
    return any([key in category_set for key in result.keys()])


def get_video_activity_recognition_categories(ml_operator: OperatorParallel, block_records: List[Record],
                                              one_category: list) -> list:
    video_ar_value_recordIDs = {}
    video_ar_value_count = {}
    video_ar_categories = []
    for record in block_records:
        video_ar_record_result = ml_operator.get_result(record)
        if video_ar_record_result in video_ar_value_count.keys():
            video_ar_value_recordIDs[video_ar_record_result].append(record.id)
            video_ar_value_count[video_ar_record_result] += 1
        else:
            video_ar_value_recordIDs[video_ar_record_result] = [record.id]
            video_ar_value_count[video_ar_record_result] = 1
    frequent_records = set()
    video_ar_categories.append(one_category)
    for item in one_category:
        if item in video_ar_value_count.keys():
            frequent_records = frequent_records.union(set(video_ar_value_recordIDs[item]))
            del video_ar_value_count[item]
    sorted_video_ar_value_count = dict(
        sorted(video_ar_value_count.items(), key=operator.itemgetter(1), reverse=True))

    for key, value in sorted_video_ar_value_count.items():
        if len(frequent_records) < 0.95 * len(block_records):
            video_ar_categories.append([key])
            frequent_records = frequent_records.union(set(video_ar_value_recordIDs[key]))
        else:
            break
    return video_ar_categories


def get_video_activity_recognition_category_label(ml_operator: OperatorParallel, record: Record, category: list) -> bool:
    category_set = set(category)
    result = ml_operator.get_result(record)
    return result in category_set


def get_ml_operator_categories(ml_operator: OperatorParallel, block_records: List[Record],
                               one_category: list) -> List:
    if isinstance(ml_operator, POSTaggerStanfordNLP):
        return get_pos_tagger_stanfordnlp_categories(ml_operator=ml_operator, block_records=block_records,
                                                     one_category=one_category)
    elif isinstance(ml_operator, SentimentStanfordNLP):
        return get_sentiment_stanfordnlp_categories(ml_operator=ml_operator, block_records=block_records,
                                                    one_category=one_category)
    elif isinstance(ml_operator, EntityStanfordNLP):
        return get_entity_stanfordnlp_categories(ml_operator=ml_operator, block_records=block_records,
                                                 one_category=one_category)
    elif isinstance(ml_operator, POSTaggerSpacyLG) or isinstance(ml_operator, POSTaggerSpacySM):
        return get_pos_tagger_spacy_categories(ml_operator=ml_operator, block_records=block_records,
                                               one_category=one_category)
    elif isinstance(ml_operator, ImageObjectDetectionYolov3):
        return get_image_object_detection_categories(ml_operator=ml_operator, block_records=block_records,
                                                     one_category=one_category)
    elif isinstance(ml_operator, VideoObjectDetectionYolov3):
        return get_video_object_detection_categories(ml_operator=ml_operator, block_records=block_records,
                                                     one_category=one_category)
    elif isinstance(ml_operator, VideoActivityRecognition):
        return get_video_activity_recognition_categories(ml_operator=ml_operator, block_records=block_records,
                                                         one_category=one_category)
    else:
        print("ml_operator is out of range")
        exit(1)


def get_record_category_label(ml_operator: OperatorParallel, record: Record, category: list) -> bool:
    if isinstance(ml_operator, POSTaggerStanfordNLP):
        return get_pos_tagger_stanfordnlp_category_label(ml_operator=ml_operator, record=record, category=category)
    elif isinstance(ml_operator, SentimentStanfordNLP):
        return get_sentiment_stanfordnlp_category_label(ml_operator=ml_operator, record=record, category=category)
    elif isinstance(ml_operator, EntityStanfordNLP):
        return get_entity_stanfordnlp_category_label(ml_operator=ml_operator, record=record, category=category)
    elif isinstance(ml_operator, POSTaggerSpacyLG) or isinstance(ml_operator, POSTaggerSpacySM):
        return get_pos_tagger_spacy_category_label(ml_operator=ml_operator, record=record, category=category)
    elif isinstance(ml_operator, ImageObjectDetectionYolov3):
        return get_image_object_detection_category_label(ml_operator=ml_operator, record=record, category=category)
    elif isinstance(ml_operator, VideoObjectDetectionYolov3):
        return get_video_object_detection_category_label(ml_operator=ml_operator, record=record, category=category)
    elif isinstance(ml_operator, VideoActivityRecognition):
        return get_video_activity_recognition_category_label(ml_operator=ml_operator, record=record, category=category)
    else:
        print("ml_operator is out of range")
        exit(1)


def compute_pair_correlation_score(block_records: List[Record], ml_operator1: OperatorParallel,
                                   one_category1: list, ml_operator2: OperatorParallel, one_category2: list) -> float:
    categories1 = get_ml_operator_categories(ml_operator=ml_operator1, block_records=block_records,
                                             one_category=one_category1)
    categories2 = get_ml_operator_categories(ml_operator=ml_operator2, block_records=block_records,
                                             one_category=one_category2)
    X = 0
    for category1 in categories1:
        for category2 in categories2:
            n_i, n_j, n_ij = 0, 0, 0
            for record in block_records:
                flag1 = get_record_category_label(ml_operator=ml_operator1, record=record, category=category1)
                flag2 = get_record_category_label(ml_operator=ml_operator2, record=record, category=category2)
                if flag1:
                    n_i += 1
                if flag2:
                    n_j += 1
                if flag1 and flag2:
                    n_ij += 1
            if n_i == 0 or n_j == 0:
                continue
            X_ij = (n_ij - n_i * n_j) * (n_ij - n_i * n_j) / (n_i * n_j)
            X += X_ij
    if min(len(categories1), len(categories2)) == 1:
        return 1000000
    else:
        return X / len(block_records) / (min(len(categories1), len(categories2)) - 1)
