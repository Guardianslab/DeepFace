import os, glob
from xperty.lib_identification import load_db_vectors
from xperty.lib_det import detector_build, face_detect
from xperty.lib_identification import recognitor_build, face_identification


os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


######### 1. Load the representations from the pickle file ###############
db_dataframe = load_db_vectors('./tmp_data/ds_model_arcface_detector_yolov8_aligned_normalization_base_expand_0_800.pkl')




######### 2. model define ###############
backends = ['opencv', 'ssd', 'dlib', 'mtcnn', 'fastmtcnn',
  'retinaface', 'mediapipe', 'yolov8', 'yunet', 'centerface',]
models = ["VGG-Face", "Facenet", "Facenet512", "OpenFace", "DeepFace", 
          "DeepID", "ArcFace", "Dlib", "SFace", "GhostFaceNet",]

backend = backends[7]
model = models[6]

face_detector = detector_build(backend_str=backend)
face_recognitor = recognitor_build(model_str=model)




######### 3. image load ###############
from deepface.commons import image_utils
file_path = './tmp_data/Ahn_Hyo_Seop_1_Ori.jpg'
img, img_name = image_utils.load_image(file_path)




######### 4. forward ###############
arg_enforce_detection = True
arg_expand_percentage = 0
arg_align = True
arg_normalization = "base" # Default is base. Options: base, raw, Facenet, Facenet2018, VGGFace, VGGFace2, ArcFace
arg_distance_metric = "cosine" # Options: 'cosine', 'euclidean', 'euclidean_l2'.
arg_threshold = 0.50

detect_result = face_detect(img, img_name, detector=face_detector, align=arg_align, expand_percentage=arg_expand_percentage, enforce_detection=arg_enforce_detection)
identification_result = face_identification(resp_objs=detect_result, df=db_dataframe, face_recognitor=face_recognitor, enforce_detection=arg_enforce_detection, align=arg_align, normalization=arg_normalization, distance_metric=arg_distance_metric, threshold=arg_threshold)

print(file_path)
print(identification_result)