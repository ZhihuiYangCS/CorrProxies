import rootpath

rootpath.append()

# the constant names of text ml operators
POSTaggerStanfordNLPON = 'POSTaggerStanfordNLP'
SentimentStanfordNLPON = 'SentimentStanfordNLP'
EntityStanfordNLPON = 'EntityStanfordNLP'
POSTaggerSpacyLGON = 'POSTaggerSpacyLG'
POSTaggerSpacySMON = 'POSTaggerSpacySM'
ImageObjectDetectionYolov3ON = 'ImageObjectDetectionYolov3'
VideoObjectDetectionYolov3ON = 'VideoObjectDetectionYolov3'
VideoActivityRecognitionON = 'VideoActivityRecognition'

"""
scheme constants
"""
SCHEME_RAW = 0
SCHEME_NS = 1
SCHEME_PP = 2
SCHEME_CORE = 3
SCHEME_COREa = 4
SCHEME_COREh = 5
SCHEME_REORDER = 6

"""
workflow_type constants
0:  text
1:  image
2:  video
"""
TWITTER_TYPE = 0
COCO_TYPE = 1
UCF101_TYPE = 2

"""
PP training method
pp_type: svm --> 0
         dnn --> 1
"""
TWITTER_PP_TYPE = TWITTER_TYPE
COCO_PP_TYPE = COCO_TYPE
UCF101_PP_TYPE = UCF101_TYPE

SVM_TRAIN_COST = 27
SVM_INFER_COST = 0.0001
DNN_TRAIN_COST = 18
DNN_INFER_COST = 0.00004

# the block size
# TWITTER_BATCH_SIZE = 1000
TWITTER_BATCH_SIZE = 500
COCO_BATCH_SIZE = 100
UCF101_BATCH_SIZE = 100

TWITTER_EXE_BATCH_SIZE = 2000
COCO_EXE_BATCH_SIZE = 1000
UCF101_EXE_BATCH_SIZE = 400

# the sample size to train filter
# TWITTER_SAMPLE_SIZE = 3000
# TWITTER_SAMPLE_COPY_SIZE = 5000
TWITTER_SAMPLE_SIZE = 2000
TWITTER_SAMPLE_COPY_SIZE = 2000

# COCO_SAMPLE_SIZE = 1000
# COCO_SAMPLE_COPY_SIZE = 3000
COCO_SAMPLE_SIZE = 400
COCO_SAMPLE_COPY_SIZE = 600

# UCF101_SAMPLE_SIZE = 400
# UCF101_SAMPLE_COPY_SIZE = 1200
UCF101_SAMPLE_SIZE = 400
UCF101_SAMPLE_COPY_SIZE = 600

# the sample size to estimate selectivity
# TWITTER_SELECT_SIZE = 1000
TWITTER_SELECT_SIZE = 1000
COCO_SELECT_SIZE = 200
UCF101_SELECT_SIZE = 100


# Total data size
TWITTER_NUM = 2000000
COCO_NUM = 123287
UCF101_NUM = 13320
# COCO_NUM = 3400
# UCF101_NUM = 1000

# the start_num to estimate execution cost and accuracy
TWITTER_START_EXEC_NUM = 1995000
COCO_START_EXEC_NUM = 122287
UCF101_START_EXEC_NUM = 12320

# selected predicates to construct workflows
POS_TAGGER_PREDICATES = [['NNPS', 'VBP', 'WRB', '.'], ['NNP', 'NNPS', 'VBD', 'VBN', 'WRB'], ['VBD', 'WRB', 'IN'],
                         ['NNPS', 'VBP', 'VBG', 'VBD', 'VBN'], ['NNPS', 'VBD', 'VBN', 'WRB', 'DT'],
                         ['NNPS', 'VB', 'VBZ', 'WRB'], ['NNPS', 'VB', 'VBD', 'VBN'], ['VBD', 'WRB', 'PRP'],
                         ['VBP', 'VBZ', 'WRB'], ['PRP']]
SENTIMENT_PREDICATES = [[1], [2], [3], [1, 2], [2, 3]]
ENTITY_PREDICATES = [['URL', 'PERSON', 'ORGANIZATION'], ['URL', 'NUMBER', 'ORGANIZATION', 'URL'],
                     ['HANDLE', 'NUMBER', 'STATE_OR_PROVINCE', 'ORGANIZATION', 'HANDLE', 'PERSON'],
                     ['HANDLE', 'PERSON', 'DATE', 'NUMBER', 'PERSON', 'HANDLE'],
                     ['HANDLE', 'NUMBER', 'STATE_OR_PROVINCE', 'ORGANIZATION', 'HANDLE']]
IMAGE_OD_PREDICATES_PERSON = [['person'], ['person', 'bowl'], ['person', 'bottle'], ['person', 'cup'],
                              ['person', 'chair'],
                              ['person', 'car'], ['person', 'dining table'], ['person', 'bottle', 'bowl'],
                              ['person', 'cup', 'bowl'], ['person', 'cup', 'bottle'], ['person', 'chair', 'bottle'],
                              ['person', 'chair', 'bowl'], ['person', 'chair', 'cup'],
                              ['person', 'dining table', 'bowl'],
                              ['person', 'car', 'bottle'], ['person', 'car', 'bowl'], ['person', 'dining table', 'cup'],
                              ['person', 'car', 'cup'], ['person', 'car', 'chair'], ['person', 'chair', 'dining table'],
                              ['person', 'dining table', 'bottle'], ['person', 'cup', 'bottle', 'bowl'],
                              ['person', 'dining table', 'cup', 'bowl'], ['person', 'chair', 'bottle', 'bowl'],
                              ['person', 'chair', 'cup', 'bowl'], ['person', 'chair', 'cup', 'bottle'],
                              ['person', 'dining table', 'bottle', 'bowl'], ['person', 'car', 'dining table'],
                              ['person', 'dining table', 'cup', 'bottle'], ['person', 'chair', 'dining table', 'bowl'],
                              ['person', 'chair', 'dining table', 'cup'], ['person', 'car', 'cup', 'bowl'],
                              ['person', 'car', 'bottle', 'bowl'], ['person', 'car', 'cup', 'bottle'],
                              ['person', 'chair', 'dining table', 'bottle'],
                              ['person', 'dining table', 'cup', 'bottle', 'bowl'],
                              ['person', 'car', 'chair', 'bottle'], ['person', 'car', 'chair', 'bowl'],
                              ['person', 'chair', 'cup', 'bottle', 'bowl'],
                              ['person', 'chair', 'dining table', 'cup', 'bowl'],
                              ['person', 'car', 'chair', 'cup'], ['person', 'car', 'dining table', 'bowl'],
                              ['person', 'car', 'dining table', 'cup'],
                              ['person', 'chair', 'dining table', 'bottle', 'bowl'],
                              ['person', 'chair', 'dining table', 'cup', 'bottle'],
                              ['person', 'car', 'chair', 'dining table'],
                              ['person', 'car', 'dining table', 'bottle'], ['person', 'car', 'cup', 'bottle', 'bowl'],
                              ['person', 'chair', 'dining table', 'cup', 'bottle', 'bowl'],
                              ['person', 'car', 'dining table', 'cup', 'bowl'],
                              ['person', 'car', 'chair', 'bottle', 'bowl'],
                              ['person', 'car', 'chair', 'cup', 'bowl'], ['person', 'car', 'chair', 'cup', 'bottle'],
                              ['person', 'car', 'dining table', 'bottle', 'bowl'],
                              ['person', 'car', 'dining table', 'cup', 'bottle'],
                              ['person', 'car', 'chair', 'dining table', 'bowl'],
                              ['person', 'car', 'chair', 'dining table', 'cup'],
                              ['person', 'car', 'chair', 'dining table', 'bottle'],
                              ['person', 'car', 'dining table', 'cup', 'bottle', 'bowl'],
                              ['person', 'car', 'chair', 'cup', 'bottle', 'bowl'],
                              ['person', 'car', 'chair', 'dining table', 'cup', 'bowl'],
                              ['person', 'car', 'chair', 'dining table', 'bottle', 'bowl'],
                              ['person', 'car', 'chair', 'dining table', 'cup', 'bottle']]
IMAGE_OD_PREDICATES_NO_PERSON = [['car', 'chair', 'dining table', 'bottle', 'dog', 'bench', 'surfboard'],
                                 ['car', 'chair', 'dining table', 'bottle', 'dog', 'handbag', 'traffic light'],
                                 ['car', 'chair', 'dining table', 'umbrella', 'bench', 'bed', 'sink'],
                                 ['car', 'chair', 'dining table', 'sink', 'cell phone', 'cat', 'bird'],
                                 ['car', 'chair', 'dining table', 'bottle', 'dog', 'pizza', 'bird'],
                                 ['car', 'chair', 'dining table', 'tv', 'sink', 'bench', 'bed'],
                                 ['car', 'chair', 'dining table', 'bowl', 'umbrella', 'bench', 'bed'],
                                 ['car', 'chair', 'dining table', 'cup', 'truck', 'dog', 'cell phone'],
                                 ['car', 'chair', 'dining table', 'handbag', 'umbrella', 'bench', 'sink'],
                                 ['car', 'chair', 'dining table', 'bowl', 'dog', 'handbag', 'bicycle'],
                                 ['car', 'chair', 'dining table', 'dog', 'tv', 'sink', 'traffic light'],
                                 ['car', 'chair', 'dining table', 'bottle', 'couch', 'bed', 'sink'],
                                 ['car', 'chair', 'cup', 'bottle', 'dog', 'handbag', 'potted plant'],
                                 ['car', 'chair', 'dining table', 'bowl', 'book', 'sink', 'surfboard'],
                                 ['car', 'chair', 'dining table', 'cup', 'cat', 'umbrella', 'motorcycle'],
                                 ['car', 'chair', 'dining table', 'sink', 'traffic light', 'handbag', 'surfboard'],
                                 ['car', 'chair', 'cup', 'bowl', 'dog', 'tv', 'surfboard'],
                                 ['car', 'chair', 'dining table', 'cup', 'bus', 'cat', 'bird'],
                                 ['car', 'chair', 'dining table', 'bottle', 'bench', 'bus', 'potted plant'],
                                 ['car', 'chair', 'dining table', 'bed', 'cat', 'traffic light', 'sink'],
                                 ['car', 'chair', 'dining table', 'bowl', 'sink', 'cat', 'vase'],
                                 ['car', 'chair', 'dining table', 'bench', 'bed', 'bird', 'vase'],
                                 ['car', 'chair', 'cup', 'bowl', 'sink', 'umbrella', 'bed'],
                                 ['car', 'chair', 'dining table', 'cup', 'dog', 'potted plant', 'traffic light'],
                                 ['car', 'chair', 'dining table', 'bowl', 'book', 'bench', 'bird'],
                                 ['car', 'chair', 'dining table', 'tv', 'sink', 'bench', 'bicycle'],
                                 ['car', 'chair', 'dining table', 'bowl', 'umbrella', 'potted plant', 'cat'],
                                 ['car', 'chair', 'dining table', 'dog', 'handbag', 'bicycle', 'vase'],
                                 ['car', 'chair', 'bottle', 'bowl', 'truck', 'cat', 'bench'],
                                 ['car', 'chair', 'cup', 'dog', 'handbag', 'sink', 'pizza'],
                                 ['car', 'chair', 'dining table', 'bed', 'bench', 'handbag', 'bicycle'],
                                 ['car', 'chair', 'cup', 'bottle', 'couch', 'handbag', 'cat'],
                                 ['car', 'chair', 'dining table', 'bottle', 'truck', 'bed', 'pizza'],
                                 ['car', 'chair', 'cup', 'bottle', 'cat', 'traffic light', 'handbag'],
                                 ['car', 'chair', 'cup', 'bottle', 'cell phone', 'cat', 'bicycle'],
                                 ['car', 'chair', 'dining table', 'dog', 'book', 'bus', 'cell phone'],
                                 ['car', 'chair', 'dining table', 'bed', 'book', 'bus', 'sink'],
                                 ['car', 'dining table', 'dog', 'bench', 'bed', 'umbrella', 'sink'],
                                 ['car', 'chair', 'dining table', 'truck', 'tv', 'umbrella', 'cat'],
                                 ['car', 'chair', 'dining table', 'bowl', 'sink', 'surfboard', 'cake'],
                                 ['car', 'chair', 'dining table', 'dog', 'sink', 'traffic light', 'pizza'],
                                 ['car', 'chair', 'dining table', 'traffic light', 'potted plant', 'cat', 'sink'],
                                 ['car', 'chair', 'cup', 'bottle', 'bed', 'cell phone', 'motorcycle'],
                                 ['car', 'chair', 'cup', 'bowl', 'truck', 'handbag', 'surfboard'],
                                 ['car', 'chair', 'bottle', 'dog', 'bench', 'cell phone', 'pizza'],
                                 ['car', 'chair', 'cup', 'bowl', 'tv', 'handbag', 'surfboard'],
                                 ['car', 'chair', 'dining table', 'truck', 'bed', 'bench', 'bicycle'],
                                 ['car', 'chair', 'dining table', 'cup', 'cat', 'book', 'motorcycle'],
                                 ['car', 'chair', 'cup', 'tv', 'bed', 'bench', 'sink'],
                                 ['car', 'chair', 'dining table', 'umbrella', 'cat', 'book', 'motorcycle'],
                                 ['car', 'chair', 'dining table', 'bottle', 'couch', 'bus', 'book'],
                                 ['car', 'chair', 'dining table', 'cup', 'bowl', 'bus', 'handbag'],
                                 ['car', 'chair', 'cup', 'bowl', 'bench', 'pizza', 'surfboard'],
                                 ['car', 'chair', 'dining table', 'dog', 'bed', 'cake', 'bicycle'],
                                 ['car', 'chair', 'dining table', 'cup', 'book', 'cat', 'bus'],
                                 ['car', 'chair', 'bottle', 'bowl', 'handbag', 'book', 'bird'],
                                 ['car', 'chair', 'dining table', 'bowl', 'couch', 'bus', 'cat'],
                                 ['car', 'dining table', 'bottle', 'couch', 'cell phone', 'umbrella', 'bed'],
                                 ['car', 'chair', 'dining table', 'book', 'surfboard', 'bird', 'vase'],
                                 ['car', 'chair', 'cup', 'handbag', 'cat', 'bench', 'cake']]

VIDEO_OD_PREDICATES = [
    ['chair', 'sports ball', 'dog', 'car', 'tv', 'horse', 'bicycle', 'skateboard', 'tennis racket', 'surfboard',
     'bird'],
    ['chair', 'sports ball', 'dog', 'car', 'tv', 'horse', 'bicycle', 'skateboard', 'boat', 'surfboard', 'cell phone'],
    ['chair', 'sports ball', 'dog', 'car', 'tv', 'horse', 'bicycle', 'skateboard', 'boat', 'bowl', 'cell phone'],
    ['chair', 'sports ball', 'dog', 'car', 'tv', 'horse', 'skateboard', 'tennis racket', 'boat', 'baseball bat',
     'toothbrush'],
    ['chair', 'sports ball', 'dog', 'car', 'tv', 'horse', 'skateboard', 'tennis racket', 'boat', 'surfboard', 'bottle'],
    ['chair', 'sports ball', 'dog', 'car', 'tv', 'horse', 'bicycle', 'skateboard', 'boat', 'dining table', 'bottle'],
    ['chair', 'sports ball', 'dog', 'car', 'tv', 'horse', 'bicycle', 'skateboard', 'boat', 'surfboard', 'bowl'],
    ['chair', 'sports ball', 'dog', 'car', 'tv', 'horse', 'bicycle', 'skateboard', 'tennis racket', 'boat', 'cup'],
    ['chair', 'sports ball', 'dog', 'car', 'tv', 'horse', 'bicycle', 'skateboard', 'boat', 'baseball bat',
     'cell phone'],
    ['chair', 'sports ball', 'dog', 'car', 'tv', 'horse', 'bicycle', 'skateboard', 'boat', 'toothbrush', 'cell phone'],
    ['chair', 'sports ball', 'dog', 'car', 'tv', 'horse', 'bicycle', 'tennis racket', 'boat', 'dining table',
     'toothbrush'],
    ['chair', 'sports ball', 'dog', 'car', 'tv', 'horse', 'skateboard', 'tennis racket', 'boat', 'dining table',
     'cell phone'],
    ['chair', 'sports ball', 'dog', 'car', 'tv', 'horse', 'skateboard', 'tennis racket', 'boat', 'toothbrush',
     'bottle'],
    ['chair', 'sports ball', 'dog', 'car', 'tv', 'horse', 'bicycle', 'skateboard', 'boat', 'baseball bat', 'surfboard'],
    ['chair', 'sports ball', 'dog', 'car', 'tv', 'horse', 'skateboard', 'tennis racket', 'boat', 'toothbrush', 'bowl'],
    ['chair', 'sports ball', 'dog', 'car', 'tv', 'horse', 'bicycle', 'skateboard', 'boat', 'baseball bat', 'bowl'],
    ['chair', 'sports ball', 'dog', 'car', 'tv', 'horse', 'bicycle', 'skateboard', 'boat', 'bottle', 'cell phone'],
    ['chair', 'sports ball', 'dog', 'car', 'tv', 'horse', 'skateboard', 'tennis racket', 'boat', 'dining table',
     'surfboard'],
    ['chair', 'sports ball', 'dog', 'car', 'tv', 'horse', 'bicycle', 'skateboard', 'tennis racket', 'boat', 'bench'],
    ['chair', 'sports ball', 'dog', 'car', 'tv', 'horse', 'bicycle', 'skateboard', 'dining table', 'surfboard',
     'toothbrush'],
    ['chair', 'sports ball', 'dog', 'car', 'tv', 'horse', 'skateboard', 'tennis racket', 'boat', 'surfboard',
     'toothbrush'],
    ['chair', 'sports ball', 'dog', 'car', 'tv', 'horse', 'bicycle', 'skateboard', 'tennis racket', 'boat', 'laptop'],
    ['chair', 'sports ball', 'dog', 'car', 'tv', 'horse', 'bicycle', 'skateboard', 'tennis racket', 'baseball bat',
     'surfboard'],
    ['chair', 'sports ball', 'dog', 'car', 'tv', 'horse', 'bicycle', 'skateboard', 'boat', 'dining table',
     'cell phone'],
    ['chair', 'sports ball', 'dog', 'car', 'tv', 'horse', 'bicycle', 'skateboard', 'boat', 'surfboard', 'bottle'],
    ['chair', 'sports ball', 'dog', 'car', 'tv', 'horse', 'bicycle', 'skateboard', 'tennis racket', 'dining table',
     'toothbrush'],
    ['chair', 'sports ball', 'dog', 'car', 'tv', 'horse', 'bicycle', 'skateboard', 'boat', 'dining table', 'surfboard'],
    ['chair', 'sports ball', 'dog', 'car', 'tv', 'horse', 'bicycle', 'skateboard', 'boat', 'baseball bat', 'bottle'],
    ['chair', 'sports ball', 'dog', 'car', 'tv', 'horse', 'bicycle', 'skateboard', 'tennis racket', 'surfboard',
     'bowl'],
    ['chair', 'sports ball', 'dog', 'car', 'tv', 'horse', 'bicycle', 'skateboard', 'boat', 'toothbrush', 'bottle'],
    ['chair', 'sports ball', 'dog', 'car', 'tv', 'horse', 'bicycle', 'skateboard', 'tennis racket', 'surfboard',
     'cell phone'],
    ['chair', 'sports ball', 'dog', 'car', 'tv', 'horse', 'bicycle', 'skateboard', 'boat', 'dining table',
     'baseball bat'],
    ['chair', 'sports ball', 'dog', 'car', 'tv', 'horse', 'bicycle', 'skateboard', 'tennis racket', 'boat', 'bird'],
    ['chair', 'sports ball', 'dog', 'car', 'tv', 'horse', 'bicycle', 'skateboard', 'boat', 'surfboard', 'toothbrush'],
    ['chair', 'sports ball', 'dog', 'car', 'tv', 'horse', 'bicycle', 'skateboard', 'boat', 'toothbrush', 'bowl'],
    ['chair', 'sports ball', 'dog', 'car', 'tv', 'horse', 'skateboard', 'tennis racket', 'boat', 'dining table',
     'toothbrush'],
    ['chair', 'sports ball', 'dog', 'car', 'tv', 'horse', 'bicycle', 'skateboard', 'boat', 'baseball bat',
     'toothbrush'],
    ['chair', 'sports ball', 'dog', 'car', 'tv', 'horse', 'bicycle', 'skateboard', 'tennis racket', 'surfboard',
     'bottle'],
    ['chair', 'sports ball', 'dog', 'car', 'tv', 'horse', 'bicycle', 'skateboard', 'tennis racket', 'dining table',
     'surfboard'],
    ['chair', 'sports ball', 'dog', 'car', 'tv', 'horse', 'bicycle', 'skateboard', 'boat', 'dining table',
     'toothbrush'],
    ['chair', 'sports ball', 'dog', 'car', 'tv', 'horse', 'bicycle', 'skateboard', 'tennis racket', 'surfboard',
     'toothbrush'],
    ['chair', 'sports ball', 'dog', 'car', 'tv', 'horse', 'bicycle', 'skateboard', 'tennis racket', 'boat',
     'baseball bat'],
    ['chair', 'sports ball', 'dog', 'car', 'tv', 'horse', 'bicycle', 'skateboard', 'tennis racket', 'boat', 'bowl'],
    ['chair', 'sports ball', 'dog', 'car', 'tv', 'horse', 'bicycle', 'skateboard', 'tennis racket', 'boat',
     'cell phone'],
    ['chair', 'sports ball', 'dog', 'car', 'tv', 'horse', 'bicycle', 'skateboard', 'tennis racket', 'boat',
     'surfboard'],
    ['chair', 'sports ball', 'dog', 'car', 'tv', 'horse', 'bicycle', 'skateboard', 'tennis racket', 'boat', 'bottle'],
    ['chair', 'sports ball', 'dog', 'car', 'tv', 'horse', 'bicycle', 'skateboard', 'tennis racket', 'boat',
     'dining table'],
    ['chair', 'sports ball', 'dog', 'car', 'tv', 'horse', 'bicycle', 'skateboard', 'tennis racket', 'boat',
     'toothbrush']]

VIDEO_AR_PREDICATES = [
    ['Archery', 'BalanceBeam', 'Basketball', 'BandMarching', 'BasketballDunk', 'BlowingCandles', 'BodyWeightSquats',
     'BreastStroke', 'BrushingTeeth', 'BaseballPitch', 'BoxingPunchingBag', 'BoxingSpeedBag', 'BabyCrawling',
     'ApplyLipstick'],
    ['Archery', 'BalanceBeam', 'Basketball', 'BandMarching', 'BasketballDunk', 'BlowingCandles', 'BodyWeightSquats',
     'BreastStroke', 'BrushingTeeth', 'BaseballPitch', 'BenchPress', 'BoxingPunchingBag', 'BabyCrawling'],
    ['Archery', 'BalanceBeam', 'Basketball', 'BandMarching', 'BasketballDunk', 'Biking', 'BodyWeightSquats',
     'BreastStroke', 'BrushingTeeth', 'BenchPress', 'BlowDryHair', 'BoxingSpeedBag', 'ApplyLipstick'],
    ['Archery', 'BalanceBeam', 'BandMarching', 'BasketballDunk', 'BlowingCandles', 'Biking', 'BodyWeightSquats',
     'BaseballPitch', 'BoxingPunchingBag', 'BlowDryHair', 'BoxingSpeedBag', 'Bowling', 'BabyCrawling', 'ApplyLipstick'],
    ['Archery', 'Basketball', 'BandMarching', 'BasketballDunk', 'BlowingCandles', 'BodyWeightSquats', 'BreastStroke',
     'BaseballPitch', 'BenchPress', 'BoxingPunchingBag', 'BlowDryHair', 'Bowling', 'BabyCrawling', 'ApplyLipstick'],
    ['BalanceBeam', 'Basketball', 'BandMarching', 'BasketballDunk', 'BlowingCandles', 'Biking', 'BodyWeightSquats',
     'BreastStroke', 'BrushingTeeth', 'BaseballPitch', 'BoxingPunchingBag', 'BlowDryHair', 'BoxingSpeedBag'],
    ['Archery', 'BalanceBeam', 'Basketball', 'BlowingCandles', 'Biking', 'BodyWeightSquats', 'BreastStroke',
     'BrushingTeeth', 'BenchPress', 'BoxingPunchingBag', 'BoxingSpeedBag', 'BabyCrawling', 'ApplyLipstick'],
    ['Archery', 'BalanceBeam', 'Basketball', 'BandMarching', 'BasketballDunk', 'Biking', 'BreastStroke', 'BenchPress',
     'BoxingPunchingBag', 'BlowDryHair', 'Bowling', 'BabyCrawling', 'ApplyLipstick'],
    ['Archery', 'Basketball', 'BandMarching', 'BasketballDunk', 'Biking', 'BodyWeightSquats', 'BreastStroke',
     'BrushingTeeth', 'BenchPress', 'BoxingPunchingBag', 'BoxingSpeedBag', 'Bowling', 'ApplyLipstick'],
    ['BalanceBeam', 'Basketball', 'BandMarching', 'BasketballDunk', 'BlowingCandles', 'Biking', 'BreastStroke',
     'BrushingTeeth', 'BaseballPitch', 'BoxingPunchingBag', 'BlowDryHair', 'BoxingSpeedBag', 'Bowling'],
    ['Archery', 'BalanceBeam', 'Basketball', 'BlowingCandles', 'Biking', 'BodyWeightSquats', 'BreastStroke',
     'BrushingTeeth', 'BenchPress', 'BoxingPunchingBag', 'BlowDryHair', 'BoxingSpeedBag'],
    ['BalanceBeam', 'Basketball', 'BandMarching', 'BasketballDunk', 'BlowingCandles', 'BodyWeightSquats',
     'BreastStroke', 'BrushingTeeth', 'BaseballPitch', 'BenchPress', 'BoxingSpeedBag', 'Bowling', 'ApplyLipstick'],
    ['BalanceBeam', 'Basketball', 'BandMarching', 'BlowingCandles', 'Biking', 'BodyWeightSquats', 'BreastStroke',
     'BrushingTeeth', 'BaseballPitch', 'BenchPress', 'BoxingSpeedBag', 'BabyCrawling', 'ApplyLipstick'],
    ['Archery', 'BalanceBeam', 'BandMarching', 'BasketballDunk', 'BlowingCandles', 'BodyWeightSquats', 'BreastStroke',
     'BrushingTeeth', 'BenchPress', 'BlowDryHair', 'BoxingSpeedBag', 'BabyCrawling'],
    ['Archery', 'BalanceBeam', 'Basketball', 'BasketballDunk', 'BlowingCandles', 'Biking', 'BreastStroke',
     'BaseballPitch', 'BenchPress', 'BoxingSpeedBag', 'BabyCrawling', 'ApplyLipstick'],
    ['Archery', 'BalanceBeam', 'Basketball', 'BandMarching', 'Biking', 'BreastStroke', 'BrushingTeeth', 'BaseballPitch',
     'BoxingPunchingBag', 'BoxingSpeedBag', 'Bowling', 'ApplyLipstick'],
    ['Archery', 'BalanceBeam', 'Basketball', 'BandMarching', 'BasketballDunk', 'BodyWeightSquats', 'BreastStroke',
     'BoxingPunchingBag', 'BlowDryHair', 'BoxingSpeedBag', 'BabyCrawling', 'ApplyLipstick'],
    ['Archery', 'BandMarching', 'BasketballDunk', 'Biking', 'BodyWeightSquats', 'BreastStroke', 'BrushingTeeth',
     'BaseballPitch', 'BlowDryHair', 'BoxingSpeedBag', 'Bowling', 'BabyCrawling', 'ApplyLipstick'],
    ['Archery', 'BalanceBeam', 'Basketball', 'BandMarching', 'Biking', 'BodyWeightSquats', 'BreastStroke',
     'BrushingTeeth', 'BaseballPitch', 'Bowling', 'BabyCrawling'],
    ['Archery', 'BalanceBeam', 'Basketball', 'BandMarching', 'BlowingCandles', 'Biking', 'BreastStroke',
     'BrushingTeeth', 'BenchPress', 'BlowDryHair', 'ApplyLipstick'],
    ['Archery', 'BalanceBeam', 'BandMarching', 'BasketballDunk', 'Biking', 'BodyWeightSquats', 'BaseballPitch',
     'BenchPress', 'BoxingPunchingBag', 'BlowDryHair', 'BoxingSpeedBag', 'BabyCrawling'],
    ['BalanceBeam', 'Basketball', 'BandMarching', 'BasketballDunk', 'BlowingCandles', 'Biking', 'BreastStroke',
     'BrushingTeeth', 'BaseballPitch', 'BlowDryHair', 'BabyCrawling', 'ApplyLipstick'],
    ['Archery', 'BalanceBeam', 'Basketball', 'BasketballDunk', 'BlowingCandles', 'BreastStroke', 'BaseballPitch',
     'BoxingPunchingBag', 'BlowDryHair', 'Bowling', 'BabyCrawling', 'ApplyLipstick'],
    ['Archery', 'BalanceBeam', 'BasketballDunk', 'BlowingCandles', 'Biking', 'BodyWeightSquats', 'BreastStroke',
     'BrushingTeeth', 'BaseballPitch', 'BoxingPunchingBag', 'BabyCrawling'],
    ['Archery', 'BalanceBeam', 'Basketball', 'BandMarching', 'BasketballDunk', 'Biking', 'BreastStroke',
     'BoxingPunchingBag', 'BlowDryHair', 'BoxingSpeedBag', 'Bowling'],
    ['Archery', 'BalanceBeam', 'BandMarching', 'BasketballDunk', 'Biking', 'BreastStroke', 'BaseballPitch',
     'BenchPress', 'BlowDryHair', 'BoxingSpeedBag', 'BabyCrawling', 'ApplyLipstick'],
    ['Archery', 'BalanceBeam', 'Basketball', 'BandMarching', 'BlowingCandles', 'Biking', 'BodyWeightSquats',
     'BlowDryHair', 'BoxingSpeedBag', 'Bowling', 'ApplyLipstick'],
    ['BalanceBeam', 'Basketball', 'BandMarching', 'BasketballDunk', 'Biking', 'BodyWeightSquats', 'BreastStroke',
     'BaseballPitch', 'BenchPress', 'BoxingSpeedBag', 'Bowling', 'BabyCrawling'],
    ['Archery', 'BalanceBeam', 'BandMarching', 'BodyWeightSquats', 'BreastStroke', 'BrushingTeeth', 'BaseballPitch',
     'BenchPress', 'BoxingPunchingBag', 'BoxingSpeedBag', 'Bowling', 'BabyCrawling'],
    ['Archery', 'BalanceBeam', 'Basketball', 'BandMarching', 'BasketballDunk', 'BodyWeightSquats', 'BaseballPitch',
     'BoxingPunchingBag', 'BlowDryHair', 'BoxingSpeedBag', 'BabyCrawling'],
    ['Archery', 'BalanceBeam', 'Basketball', 'BandMarching', 'BasketballDunk', 'BodyWeightSquats', 'BaseballPitch',
     'BenchPress', 'BoxingSpeedBag', 'Bowling', 'BabyCrawling'],
    ['Archery', 'Basketball', 'BandMarching', 'BasketballDunk', 'BlowingCandles', 'BodyWeightSquats', 'BreastStroke',
     'BrushingTeeth', 'BoxingSpeedBag', 'Bowling', 'ApplyLipstick'],
    ['Archery', 'BalanceBeam', 'Basketball', 'BasketballDunk', 'BlowingCandles', 'BodyWeightSquats', 'BrushingTeeth',
     'BoxingPunchingBag', 'BoxingSpeedBag', 'BabyCrawling', 'ApplyLipstick'],
    ['BalanceBeam', 'Basketball', 'BandMarching', 'BlowingCandles', 'Biking', 'BodyWeightSquats', 'BreastStroke',
     'BrushingTeeth', 'BaseballPitch', 'BenchPress', 'Bowling'],
    ['Archery', 'BalanceBeam', 'Basketball', 'BasketballDunk', 'BlowingCandles', 'Biking', 'BreastStroke',
     'BrushingTeeth', 'BlowDryHair', 'BoxingSpeedBag'],
    ['Archery', 'Basketball', 'BandMarching', 'BasketballDunk', 'BlowingCandles', 'Biking', 'BodyWeightSquats',
     'BlowDryHair', 'BoxingSpeedBag', 'Bowling', 'ApplyLipstick'],
    ['Archery', 'BandMarching', 'BasketballDunk', 'BlowingCandles', 'Biking', 'BodyWeightSquats', 'BreastStroke',
     'BrushingTeeth', 'BenchPress', 'Bowling', 'ApplyLipstick'],
    ['Archery', 'BalanceBeam', 'Basketball', 'BlowingCandles', 'BreastStroke', 'BrushingTeeth', 'BaseballPitch',
     'BenchPress', 'BlowDryHair', 'BoxingSpeedBag', 'Bowling'],
    ['Archery', 'BalanceBeam', 'BandMarching', 'BasketballDunk', 'Biking', 'BreastStroke', 'BaseballPitch',
     'BenchPress', 'BlowDryHair', 'BoxingSpeedBag', 'BabyCrawling'],
    ['Archery', 'Basketball', 'BandMarching', 'BasketballDunk', 'Biking', 'BodyWeightSquats', 'BreastStroke',
     'BoxingPunchingBag', 'BoxingSpeedBag', 'Bowling', 'ApplyLipstick'],
    ['Archery', 'BalanceBeam', 'BandMarching', 'BasketballDunk', 'BlowingCandles', 'Biking', 'BreastStroke',
     'BrushingTeeth', 'BlowDryHair', 'BabyCrawling'],
    ['Archery', 'Basketball', 'BlowingCandles', 'Biking', 'BodyWeightSquats', 'BreastStroke', 'BrushingTeeth',
     'BaseballPitch', 'BlowDryHair', 'Bowling', 'ApplyLipstick'],
    ['Archery', 'BalanceBeam', 'Basketball', 'BandMarching', 'BlowingCandles', 'BaseballPitch', 'BoxingPunchingBag',
     'BlowDryHair', 'BoxingSpeedBag', 'Bowling', 'BabyCrawling'],
    ['Basketball', 'BandMarching', 'BasketballDunk', 'Biking', 'BodyWeightSquats', 'BreastStroke', 'BrushingTeeth',
     'BenchPress', 'BoxingPunchingBag', 'BlowDryHair', 'BoxingSpeedBag', 'Bowling'],
    ['Archery', 'BalanceBeam', 'Basketball', 'BandMarching', 'BlowingCandles', 'BaseballPitch', 'BenchPress',
     'BlowDryHair', 'BoxingSpeedBag', 'BabyCrawling', 'ApplyLipstick'],
    ['Archery', 'BalanceBeam', 'BasketballDunk', 'BlowingCandles', 'BodyWeightSquats', 'BreastStroke', 'BaseballPitch',
     'BoxingPunchingBag', 'BoxingSpeedBag', 'Bowling', 'BabyCrawling'],
    ['Archery', 'Basketball', 'BandMarching', 'BasketballDunk', 'BlowingCandles', 'Biking', 'BrushingTeeth',
     'BaseballPitch', 'BenchPress', 'Bowling'],
    ['Archery', 'BalanceBeam', 'BandMarching', 'Biking', 'BreastStroke', 'BrushingTeeth', 'BaseballPitch', 'BenchPress',
     'BoxingSpeedBag', 'Bowling', 'ApplyLipstick'],
    ['Archery', 'BalanceBeam', 'Basketball', 'BandMarching', 'BlowingCandles', 'BreastStroke', 'BrushingTeeth',
     'BoxingPunchingBag', 'Bowling', 'BabyCrawling'],
    ['Archery', 'BalanceBeam', 'BasketballDunk', 'BlowingCandles', 'BodyWeightSquats', 'BreastStroke', 'BenchPress',
     'BoxingPunchingBag', 'BlowDryHair', 'Bowling', 'ApplyLipstick']]
# experiments
target_accuracyies = [0.9, 0.92, 0.94, 0.96, 0.98]
TWITTER_DATA_SIZES_SCALABILITY = [int(TWITTER_NUM * i / 10) for i in range(1, 11, 1)]

# experiments default values
WORKFLOW_TYPE_DEFAULT = TWITTER_TYPE  # twitter
TARGET_ACCURACY_DEFAULT = 0.9
SCHEME_DEFAULT = SCHEME_CORE  # CORE
OPT_THREAD_NUM_DEFAULT = 4
EXE_THREAD_NUM_DEFAULT = 4

"""
exhausted_cost_comp --> 0; 
exhausted_cost_func --> 1;
hill_climbing_cost_comp --> 2; 
hill_climbing_cost_func --> 3
"""
AA_METHOD_EXHAUSTED_CC = 0
AA_METHOD_EXHAUSTED_CF = 1
AA_METHOD_HCLIMBING_CC = 2
AA_METHOD_HCLIMBING_CF = 3

AA_METHOD_DEFAULT = AA_METHOD_HCLIMBING_CC  # hill_climbing_cost_comp

"""
resample retrain --> 0; 
reuse sample, retrain --> 1;
reuse sample, reuse PPs --> 2
"""
AA_MODE_RS_RT = 0
AA_MODE_S_RT = 1
AA_MODE_S_T = 2

AA_MODE_DEFAULT = AA_MODE_S_T  # reuse sample, reuse PPs

"""
AccuracyAllocatorExhaustedSearch: --> 0; 
AccuracyAllocatorHillClimbing: --> 1
"""
REORDER_AA_METHOD_EXHAUSTED = 0
REORDER_AA_METHOD_HCLIMBING = 1
REORDER_AA_METHOD_DEFAULT = REORDER_AA_METHOD_HCLIMBING  # AccuracyAllocatorHillClimbing
