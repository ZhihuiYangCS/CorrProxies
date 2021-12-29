import signal
import sys
import time

import rootpath


rootpath.append()
from operators.ml_operators.image_video_operators.image_object_detection_yolov3 import ImageObjectDetectionYolov3
from operators.ml_operators.image_video_operators.video_activity_recognition import VideoActivityRecognition
from operators.ml_operators.image_video_operators.video_object_detection_yolov3 import VideoObjectDetectionYolov3
from operators.ml_operators.text_operators.spacy.spacy_operator import POSTaggerSpacyLG, POSTaggerSpacySM
from operators.ml_operators.text_operators.stanfordnlp.entity_stanfordnlp import EntityStanfordNLP
from operators.ml_operators.text_operators.stanfordnlp.pos_tagger_stanfordnlp import POSTaggerStanfordNLP
from operators.ml_operators.text_operators.stanfordnlp.sentiment_stanfordnlp import SentimentStanfordNLP
from ml_workflow.exps.WorkflowExp1 import WorkflowExp1
from ml_workflow.exps.WorkflowExp2 import WorkflowExp2
from ml_workflow.exps.WorkflowExp3 import WorkflowExp3
from ml_workflow.exps.WorkflowExp4 import WorkflowExp4
from optimizer.PPs.preprocessor.image_preprocessor import ImagePreprocessor
from optimizer.PPs.preprocessor.text_preprocessor import TextPreprocessor
from optimizer.PPs.preprocessor.video_preprocessor import VideoPreprocessor
from utility.constant import TWITTER_TYPE, COCO_TYPE
from ml_workflow.construction.generate_workflows import get_workflow

if __name__ == '__main__':
    workflow = None
    ml_operators = None
    preprocessor = None


    def signal_handler(sig, frame):
        print('You pressed Ctrl+C!')
        scan, sink = workflow.get_scan_sink()
        scan.close()
        sink.close()
        for operator in ml_operators:
            if isinstance(operator, POSTaggerStanfordNLP) or isinstance(operator, SentimentStanfordNLP) or isinstance(
                    operator, EntityStanfordNLP):
                models[operator.operator_name].close()
        sys.exit(0)


    exp_type = int(sys.argv[1])
    workflow_type = int(sys.argv[2])
    workflow_index = int(sys.argv[3])
    target_accuracy = float(sys.argv[4])
    scheme = int(sys.argv[5])
    opt_thread_num = int(sys.argv[6])
    exe_thread_num = int(sys.argv[7])
    print("exp_type = " + str(exp_type) + "; workflow_type = " + str(workflow_type) + "; workflow_index = " + str(
        workflow_index) + "; target_accuracy = " + str(target_accuracy) + "; scheme = " + str(
        scheme) + "; opt_thread_num = " + str(opt_thread_num) + "; exe_thread_num = " + str(exe_thread_num))

    signal.signal(signal.SIGINT, signal_handler)
    time1 = time.time()
    if workflow_type == TWITTER_TYPE:
        pos_tagger_stanfordnlp = POSTaggerStanfordNLP()
        sentiment_stanfordnlp = SentimentStanfordNLP()
        entity_stanfordnlp = EntityStanfordNLP()
        pos_tagger_spacyLG = POSTaggerSpacyLG()
        pos_tagger_spacySM = POSTaggerSpacySM()
        ml_operators = [pos_tagger_stanfordnlp, sentiment_stanfordnlp, entity_stanfordnlp, pos_tagger_spacyLG,
                        pos_tagger_spacySM]
        preprocessor = TextPreprocessor()
        models = {}
        for operator in ml_operators:
            models[operator.operator_name] = operator.load_model()
    elif workflow_type == COCO_TYPE:
        ml_operators = [ImageObjectDetectionYolov3()]
        preprocessor = ImagePreprocessor()
        models = {}
    else:
        ml_operators = [VideoObjectDetectionYolov3(), VideoActivityRecognition()]
        preprocessor = VideoPreprocessor()
        models = {}
    time2 = time.time()
    print("initialize ml_operators and preprocessors cost = " + str(time2 - time1))
    time3 = time.time()
    print("load ml_models cost = " + str(time3 - time2))
    operators = get_workflow(workflow_type=workflow_type, workflow_index=workflow_index, ml_operators=ml_operators)
    if exp_type == 1:
        workflow = WorkflowExp1(operators=operators, preprocessor=preprocessor, models=models,
                                workflow_name=str(workflow_index), target_accuracy=target_accuracy, scheme=scheme,
                                opt_thread_num=opt_thread_num, exe_thread_num=exe_thread_num)
    elif exp_type == 2:
        workflow = WorkflowExp2(operators=operators, preprocessor=preprocessor, models=models,
                                workflow_name=str(workflow_index), target_accuracy=target_accuracy, scheme=scheme,
                                opt_thread_num=opt_thread_num, exe_thread_num=exe_thread_num)
    elif exp_type == 3:
        workflow = WorkflowExp3(operators=operators, preprocessor=preprocessor, models=models,
                                workflow_name=str(workflow_index), target_accuracy=target_accuracy, scheme=scheme,
                                opt_thread_num=opt_thread_num, exe_thread_num=exe_thread_num)
    elif exp_type == 4:
        workflow = WorkflowExp4(operators=operators, preprocessor=preprocessor, models=models,
                                workflow_name=str(workflow_index), target_accuracy=target_accuracy, scheme=scheme,
                                opt_thread_num=opt_thread_num, exe_thread_num=exe_thread_num)
    else:
        print("exp type = " + str(exp_type) + " does not exist ...")
        sys.exit(0)
    workflow.run()
    scan, sink = workflow.get_scan_sink()
    scan.close()
    sink.close()
    for operator in ml_operators:
        if isinstance(operator, POSTaggerStanfordNLP) or isinstance(operator, SentimentStanfordNLP) or isinstance(
                operator, EntityStanfordNLP):
            models[operator.operator_name].close()
