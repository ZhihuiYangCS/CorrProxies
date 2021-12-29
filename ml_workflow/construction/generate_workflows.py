import csv
import itertools
import time
from typing import List, Union, Any

import pandas
import rootpath

rootpath.append()
from operators.ml_operators.image_video_operators.video_activity_recognition import VideoActivityRecognition
from operators.scan.image_json_scan import ImageJsonScan
from operators.scan.video_json_scan import VideoJsonScan
from operators.ml_operators.image_video_operators.image_object_detection_yolov3 import ImageObjectDetectionYolov3
from operators.ml_operators.image_video_operators.video_object_detection_yolov3 import VideoObjectDetectionYolov3
from operators.ml_operators.text_operators.spacy.spacy_operator import POSTaggerSpacyLG, POSTaggerSpacySM
from operators.ml_operators.text_operators.stanfordnlp.entity_stanfordnlp import EntityStanfordNLP
from operators.ml_operators.text_operators.stanfordnlp.pos_tagger_stanfordnlp import POSTaggerStanfordNLP
from operators.ml_operators.text_operators.stanfordnlp.sentiment_stanfordnlp import SentimentStanfordNLP
from operators.operator_base.operator_parallel import OperatorParallel
from utility.correlation_utility import read_records, compute_pair_correlation_score
from utility.constant import POS_TAGGER_PREDICATES, SENTIMENT_PREDICATES, ENTITY_PREDICATES, POSTaggerStanfordNLPON, \
    POSTaggerSpacyLGON, POSTaggerSpacySMON, EntityStanfordNLPON, SentimentStanfordNLPON, TWITTER_TYPE, COCO_TYPE, \
    VideoObjectDetectionYolov3ON, VideoActivityRecognitionON, VIDEO_OD_PREDICATES, VIDEO_AR_PREDICATES, \
    IMAGE_OD_PREDICATES_PERSON, IMAGE_OD_PREDICATES_NO_PERSON, UCF101_TYPE
from operators.ml_filter import MLFilter
from operators.ml_predicator import MLPredicator
from operators.scan.text_json_scan import TextJsonScan
from operators.sink import Sink
from paths import TEXT_WORKFLOW_PATH, TEXT_TWITTER_ALL_PATH, TEMP_TEST_PATH, CORRELATION_TWITTER_PATH, \
    IMAGE_COCO_ALL_PATH, VIDEO_UCF101_ALL_PATH, VIDEO_WORKFLOW_PATH, IMAGE_WORKFLOW_PATH, CORRELATION_UCF101_PATH, \
    CORRELATION_COCO_PATH


def get_ml_operator_domain(ml_operator_name: str) -> Union[List[List[str]], None]:
    domain = None
    if ml_operator_name == POSTaggerStanfordNLPON or ml_operator_name == POSTaggerSpacyLGON or ml_operator_name == POSTaggerSpacySMON:
        domain = POS_TAGGER_PREDICATES
    if ml_operator_name == SentimentStanfordNLPON:
        domain = SENTIMENT_PREDICATES
    if ml_operator_name == EntityStanfordNLPON:
        domain = ENTITY_PREDICATES
    if ml_operator_name == VideoObjectDetectionYolov3ON:
        domain = VIDEO_OD_PREDICATES
    if ml_operator_name == VideoActivityRecognitionON:
        domain = VIDEO_AR_PREDICATES
    return domain


def get_ml_operator(ml_operators: List[OperatorParallel], ml_operator_name: str) -> Union[OperatorParallel, None]:
    result_ml_operator = None
    for ml_operator in ml_operators:
        if ml_operator.operator_name == ml_operator_name:
            result_ml_operator = ml_operator
            # result_ml_operator = ml_operator.get_new_object()
    return result_ml_operator


def generate_text_workflows(text_ml_operators: List[OperatorParallel]):
    workflow_index = 0
    fields = ['workflow_index', 'ml_operator_name_list', 'conditions_list', 'correlation_score']
    block_records = read_records(path=CORRELATION_TWITTER_PATH)
    with open(TEXT_WORKFLOW_PATH, "w") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(fields)
        two_combination = list(itertools.combinations(text_ml_operators, 2))
        for combination in two_combination:
            combination_list = list(combination)
            domain0 = get_ml_operator_domain(ml_operator_name=combination_list[0].operator_name)
            domain1 = get_ml_operator_domain(ml_operator_name=combination_list[1].operator_name)
            for condition0 in domain0:
                for condition1 in domain1:
                    correlation_score = compute_pair_correlation_score(block_records=block_records,
                                                                       ml_operator1=combination_list[0],
                                                                       one_category1=condition0,
                                                                       ml_operator2=combination_list[1],
                                                                       one_category2=condition1)
                    csv_writer.writerow(
                        [workflow_index, [combination_list[0].operator_name, combination_list[1].operator_name],
                         [condition0, condition1], correlation_score])
                    workflow_index += 1
                    csv_writer.writerow(
                        [workflow_index, [combination_list[1].operator_name, combination_list[0].operator_name],
                         [condition1, condition0], correlation_score])
                    workflow_index += 1
                    print(str(workflow_index) + ";\t" + combination_list[0].operator_name + ";\t" + combination_list[
                        1].operator_name + ";\tcorrelation_score = " + str(correlation_score))
        three_combination = list(itertools.combinations(text_ml_operators, 3))
        for combination in three_combination:
            combination_list = list(combination)
            domain0 = get_ml_operator_domain(ml_operator_name=combination_list[0].operator_name)
            domain1 = get_ml_operator_domain(ml_operator_name=combination_list[1].operator_name)
            domain2 = get_ml_operator_domain(ml_operator_name=combination_list[2].operator_name)
            for condition0 in domain0:
                for condition1 in domain1:
                    for condition2 in domain2:
                        correlation_score = max([compute_pair_correlation_score(block_records=block_records,
                                                                                ml_operator1=combination_list[0],
                                                                                one_category1=condition0,
                                                                                ml_operator2=combination_list[1],
                                                                                one_category2=condition1),
                                                 compute_pair_correlation_score(block_records=block_records,
                                                                                ml_operator1=combination_list[0],
                                                                                one_category1=condition0,
                                                                                ml_operator2=combination_list[2],
                                                                                one_category2=condition2),
                                                 compute_pair_correlation_score(block_records=block_records,
                                                                                ml_operator1=combination_list[1],
                                                                                one_category1=condition1,
                                                                                ml_operator2=combination_list[2],
                                                                                one_category2=condition2)])
                        csv_writer.writerow([workflow_index,
                                             [combination_list[0].operator_name, combination_list[1].operator_name,
                                              combination_list[2].operator_name], [condition0, condition1, condition2],
                                             correlation_score])
                        workflow_index += 1
                        csv_writer.writerow([workflow_index,
                                             [combination_list[0].operator_name, combination_list[2].operator_name,
                                              combination_list[1].operator_name],
                                             [condition0, condition2, condition1], correlation_score])
                        workflow_index += 1
                        csv_writer.writerow([workflow_index,
                                             [combination_list[1].operator_name, combination_list[0].operator_name,
                                              combination_list[2].operator_name],
                                             [condition1, condition0, condition2], correlation_score])
                        workflow_index += 1
                        csv_writer.writerow([workflow_index,
                                             [combination_list[1].operator_name, combination_list[2].operator_name,
                                              combination_list[0].operator_name],
                                             [condition1, condition2, condition0], correlation_score])
                        workflow_index += 1
                        csv_writer.writerow([workflow_index,
                                             [combination_list[2].operator_name, combination_list[0].operator_name,
                                              combination_list[1].operator_name],
                                             [condition2, condition0, condition1], correlation_score])
                        workflow_index += 1
                        csv_writer.writerow([workflow_index,
                                             [combination_list[2].operator_name, combination_list[1].operator_name,
                                              combination_list[0].operator_name],
                                             [condition2, condition1, condition0], correlation_score])
                        workflow_index += 1
                        print(
                            str(workflow_index) + ";\t" + combination_list[0].operator_name + ";\t" + combination_list[
                                1].operator_name + ";\t" + combination_list[
                                2].operator_name + ";\tcorrelation_score = " + str(correlation_score))
    print("text workflow num = " + str(workflow_index))


def generate_image_workflows(image_object_detetction: OperatorParallel):
    workflow_index = 0
    fields = ['workflow_index', 'ml_operator_name_list', 'conditions_list', 'correlation_score']
    block_records = read_records(path=CORRELATION_COCO_PATH)
    with open(IMAGE_WORKFLOW_PATH, "w") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(fields)
        print("image object detection predicates with person domain size = " + str(len(IMAGE_OD_PREDICATES_PERSON)))
        print("image object detection predicates with person domain size = " + str(len(IMAGE_OD_PREDICATES_NO_PERSON)))
        for i in range(len(IMAGE_OD_PREDICATES_PERSON)):
            for j in range(i + 1, len(IMAGE_OD_PREDICATES_NO_PERSON)):
                time1 = time.time()
                # compute a pair of correlation score on image takes about 513.4454667568207 seconds.
                # this is because the size of category is huge, such as 61 * 60
                # so we estimate the correlation score on 1000 records.[:1000]
                correlation_score = compute_pair_correlation_score(block_records=block_records,
                                                                   ml_operator1=image_object_detetction,
                                                                   one_category1=IMAGE_OD_PREDICATES_PERSON[i],
                                                                   ml_operator2=image_object_detetction,
                                                                   one_category2=IMAGE_OD_PREDICATES_NO_PERSON[j])
                time2 = time.time()
                print("compute correlation score time = " + str(time2 - time1))
                csv_writer.writerow(
                    [workflow_index, [image_object_detetction.operator_name, image_object_detetction.operator_name],
                     [IMAGE_OD_PREDICATES_PERSON[i], IMAGE_OD_PREDICATES_NO_PERSON[j]], correlation_score])
                workflow_index += 1
                csv_writer.writerow(
                    [workflow_index, [image_object_detetction.operator_name, image_object_detetction.operator_name],
                     [IMAGE_OD_PREDICATES_NO_PERSON[j], IMAGE_OD_PREDICATES_PERSON[i]], correlation_score])
                workflow_index += 1
                print(str(
                    workflow_index) + ";\t" + image_object_detetction.operator_name + ";\t" + image_object_detetction.operator_name + ";\tcorrelation_score = " + str(
                    correlation_score))
    print("image workflow num = " + str(workflow_index))


def generate_video_workflows(video_object_detetction: OperatorParallel, video_activity_recognition: OperatorParallel):
    workflow_index = 0
    fields = ['workflow_index', 'ml_operator_name_list', 'conditions_list', 'correlation_score']
    block_records = read_records(path=CORRELATION_UCF101_PATH)
    with open(VIDEO_WORKFLOW_PATH, "w") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(fields)
        domain0 = get_ml_operator_domain(ml_operator_name=video_object_detetction.operator_name)
        domain1 = get_ml_operator_domain(ml_operator_name=video_activity_recognition.operator_name)
        print(len(domain0))
        print(len(domain1))
        for condition0 in domain0:
            for condition1 in domain1:
                correlation_score = compute_pair_correlation_score(block_records=block_records,
                                                                   ml_operator1=video_object_detetction,
                                                                   one_category1=condition0,
                                                                   ml_operator2=video_activity_recognition,
                                                                   one_category2=condition1)
                csv_writer.writerow(
                    [workflow_index, [video_object_detetction.operator_name, video_activity_recognition.operator_name],
                     [condition0, condition1], correlation_score])
                workflow_index += 1
                csv_writer.writerow(
                    [workflow_index, [video_activity_recognition.operator_name, video_object_detetction.operator_name],
                     [condition1, condition0], correlation_score])
                workflow_index += 1
                print(str(
                    workflow_index) + ";\t" + video_object_detetction.operator_name + ";\t" + video_activity_recognition.operator_name + ";\tcorrelation_score = " + str(
                    correlation_score))
    print("video workflow num = " + str(workflow_index))


def get_ml_predicate(ml_operator: OperatorParallel, condition: list, predicate_name: str) -> Union[MLPredicator, None]:
    ml_predicate = None
    if isinstance(ml_operator, POSTaggerStanfordNLP) or isinstance(ml_operator, EntityStanfordNLP):
        def func(result: Any):
            return any([item in condition for item in result])

        ml_predicate = MLPredicator(ml_operator=ml_operator, operator_name=predicate_name, predicate=func)
    elif isinstance(ml_operator, SentimentStanfordNLP):
        def func(result: Any):
            long_sentence = 0
            sentiment = 0
            for s in result:
                sentence = " ".join([t["word"] for t in s["tokens"]])
                if long_sentence < len(sentence):
                    long_sentence = len(sentence)
                    sentiment = s["sentimentValue"]
            return sentiment in condition

        ml_predicate = MLPredicator(ml_operator=ml_operator, operator_name=predicate_name, predicate=func)
    elif isinstance(ml_operator, POSTaggerSpacyLG) or isinstance(ml_operator, POSTaggerSpacySM):
        def func(result: Any):
            return any([value[0] in condition for token, value in result.items()])

        ml_predicate = MLPredicator(ml_operator=ml_operator, operator_name=predicate_name, predicate=func)
    elif isinstance(ml_operator, ImageObjectDetectionYolov3) or isinstance(ml_operator, VideoObjectDetectionYolov3):
        def func(result: Any):
            return any([key in condition for key in result.keys()])

        ml_predicate = MLPredicator(ml_operator=ml_operator, operator_name=predicate_name, predicate=func)
    elif isinstance(ml_operator, VideoActivityRecognition):
        def func(result: Any):
            return result in condition

        ml_predicate = MLPredicator(ml_operator=ml_operator, operator_name=predicate_name, predicate=func)
    return ml_predicate


def get_workflow(workflow_type: int, workflow_index: int, ml_operators: List[OperatorParallel]) -> List[OperatorParallel]:
    workflow_operators = []
    if workflow_type == TWITTER_TYPE:  # text
        workflow_operators.append(TextJsonScan(TEXT_TWITTER_ALL_PATH))
        workflows = pandas.read_csv(TEXT_WORKFLOW_PATH)
    elif workflow_type == COCO_TYPE:  # image
        workflow_operators.append(ImageJsonScan(IMAGE_COCO_ALL_PATH))
        workflows = pandas.read_csv(IMAGE_WORKFLOW_PATH)
    else:  # video
        workflow_operators.append(VideoJsonScan(VIDEO_UCF101_ALL_PATH))
        workflows = pandas.read_csv(VIDEO_WORKFLOW_PATH)
    workflow = workflows[workflows['workflow_index'].isin([workflow_index])]
    print("workflow_index = " + str(list(workflow['workflow_index'])[0]) + "; ml_operator_names = " + str(
        list(workflow['ml_operator_name_list'])[0]) + "; conditions = " + str(
        list(workflow['conditions_list'])[0]) + "; correlation_score = " + str(list(workflow['correlation_score'])[0]))
    ml_operator_name_list = list(workflow['ml_operator_name_list'])[0].strip('][').split(', ')
    conditions_list = list(workflow['conditions_list'])[0].strip('][').split('], [')
    for i in range(len(ml_operator_name_list)):
        # ml_operator name needs strip()
        if i > 0 and ml_operator_name_list[i - 1].strip('\'') == ml_operator_name_list[i].strip('\''):
            # because there is additional scan operator in workflow_operators.
            ml_operator = workflow_operators[i].ml_predicator.ml_operator
        else:
            ml_operator = get_ml_operator(ml_operators=ml_operators,
                                          ml_operator_name=ml_operator_name_list[i].strip('\''))
        condition_str = conditions_list[i].split(', ')  # after split, its type is list now
        condition_list = []
        for condition in condition_str:
            condition_list.append(condition.replace('\'', ''))
        ml_predicate = get_ml_predicate(ml_operator=ml_operator, condition=condition_list, predicate_name=str(i))
        workflow_operators.append(MLFilter(ml_predicator=ml_predicate))
    workflow_operators.append(Sink(TEMP_TEST_PATH))
    print(workflow_operators)
    return workflow_operators
