import os

# root path of the project
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(ROOT_DIR, 'data')

MODEL_DIR = os.path.join(ROOT_DIR, 'model')

OUTPUT_DIR = os.path.join(ROOT_DIR, 'output')

OPERATORS_DIR = os.path.join(ROOT_DIR, 'operators')


# data paths
TEXT_DATA_DIR = os.path.join(DATA_DIR, 'text')
IMAGE_DATA_DIR = os.path.join(DATA_DIR, 'image')
VIDEO_DATA_DIR = os.path.join(DATA_DIR, 'video')

TWITTER_DATA_DIR = os.path.join(TEXT_DATA_DIR, 'twitter')
COCO_DATA_DIR = os.path.join(IMAGE_DATA_DIR, 'coco')
UCF101_DATA_DIR = os.path.join(VIDEO_DATA_DIR, 'ucf101')

TEXT_TWITTER_TRAIN_DATA_PATH = os.path.join(TWITTER_DATA_DIR, 'train')
# TEXT_TWITTER_TRAIN_YAO_DATA_PATH = os.path.join(TWITTER_DATA_DIR, 'train_yao')
TEXT_TWITTER_VALI_DATA_PATH = os.path.join(TWITTER_DATA_DIR, 'validation')
# TEXT_TWITTER_TRAIN_DATA_PATH_1 = os.path.join(TWITTER_DATA_DIR, 'train_1')
# TEXT_TWITTER_VALI_DATA_PATH_1 = os.path.join(TWITTER_DATA_DIR, 'validation_1')
TEXT_TWITTER_ALL_PATH = os.path.join(TWITTER_DATA_DIR, 'twitter_all')

IMAGE_COCO_TRAIN_DATA_PATH = os.path.join(COCO_DATA_DIR, 'train')
# IMAGE_COCO_TRAIN_YAO_DATA_PATH = os.path.join(COCO_DATA_DIR, 'train_yao')
IMAGE_COCO_VALI_DATA_PATH = os.path.join(COCO_DATA_DIR, 'validation')
IMAGE_COCO_ALL_PATH = os.path.join(COCO_DATA_DIR, 'coco_all')

VIDEO_UCF101_TRAIN_DATA_PATH = os.path.join(UCF101_DATA_DIR, 'train')
# VIDEO_UCF101_TRAIN_YAO_DATA_PATH = os.path.join(UCF101_DATA_DIR, 'train_yao')
VIDEO_UCF101_VALI_DATA_PATH = os.path.join(UCF101_DATA_DIR, 'validation')
VIDEO_UCF101_ALL_PATH = os.path.join(UCF101_DATA_DIR, 'ucf101_all')

# VIDEO_UCF101_TRAIN_DATA_PATH_1 = os.path.join(UCF101_DATA_DIR, 'train_1')
# VIDEO_UCF101_VALI_DATA_PATH_1 = os.path.join(UCF101_DATA_DIR, 'validation_1')

COCO_IMAGE_PATH = os.path.join(COCO_DATA_DIR, 'coco', 'images')
COCO_IMAGE_CONVERT_PATH = os.path.join(COCO_DATA_DIR, 'coco', 'images_convert')

UCF101_FRAMES_PATH = os.path.join(UCF101_DATA_DIR, 'frames')
UCF101_VIDEO_PATH = os.path.join(UCF101_DATA_DIR, 'UCF-101')

TEMP_TEST_PATH = os.path.join(ROOT_DIR, 'temp_test')


# model paths
STANFORD_NLP_MODEL_PATH = os.path.join(MODEL_DIR, 'stanford-corenlp-full-2018-10-05')
OPEN_NLP_MODEL_PATH = os.path.join(MODEL_DIR, 'apache-opennlp-1.9.2')
VOCAB_PATH = os.path.join(MODEL_DIR, 'liblinear_vocab.pkl')

YOLO_DIR = os.path.join(MODEL_DIR, 'yolov3')
YOLO_WEIGHTS_PATH = os.path.join(YOLO_DIR, 'yolov3.pt')
YOLO_TINY_WEIGHTS_PATH = os.path.join(YOLO_DIR, 'yolov3-tiny.pt')

KERAS_DIR = os.path.join(MODEL_DIR, 'keras_video_classifier')
KERAS_MODEL_PATH = os.path.join(KERAS_DIR, 'demo', 'models', 'UCF-101')

RESNET152NN_DIR = os.path.join(MODEL_DIR, 'resnet152nn')
RESNET152NN_FEATURE_DIR = os.path.join(RESNET152NN_DIR, "UCF-101-Features")

# workflow paths
TEXT_WORKFLOW_PATH = DATA_DIR + '/workflows/text_workflows.csv'
IMAGE_WORKFLOW_PATH = DATA_DIR + '/workflows/image_workflows.csv'
VIDEO_WORKFLOW_PATH = DATA_DIR + '/workflows/video_workflows.csv'

# correlation analysis
CORRELATION_DATA_DIR = os.path.join(DATA_DIR, 'correlation_analysis')
CORRELATION_TWITTER_PATH = os.path.join(CORRELATION_DATA_DIR, 'text_data_prepare.json')
CORRELATION_COCO_PATH = os.path.join(CORRELATION_DATA_DIR, 'image_data_prepare.json')
CORRELATION_UCF101_PATH = os.path.join(CORRELATION_DATA_DIR, 'video_data_prepare.json')


