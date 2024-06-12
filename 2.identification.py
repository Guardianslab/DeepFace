import os, glob
from xperty.lib_identification import load_db_vectors
from xperty.lib_identification import recognitor_build, face_identification


os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


######### 1. Load the representations from the pickle file ###############
db_dataframe = load_db_vectors('./tmp_data/ds_model_arcface_detector_yolov8_aligned_normalization_base_expand_0_800.pkl')

######### 2. model define ###############
models = ["VGG-Face", "Facenet", "Facenet512", "OpenFace", "DeepFace", 
          "DeepID", "ArcFace", "Dlib", "SFace", "GhostFaceNet",]
model = models[6]

face_recognitor = recognitor_build(model_str=model)


######### 3. image load ###############
from deepface.commons import image_utils
# file_path_list = glob.glob('../20240603/*/*')
file_path = './tmp_data/Ahn_Hyo_Seop_1_Crop.jpg'
img, img_name = image_utils.load_image(file_path)
detect_result = [{
                "face": img[:, :, ::-1] / 255,
                "facial_area": {
                    "x": int(0),
                    "y": int(0),
                    "w": int(img.shape[1]),
                    "h": int(img.shape[0]),
                    "left_eye": None,
                    "right_eye": None,
                },
                "confidence": 0,
            }]


######### 4. forward ###############
arg_enforce_detection = True
arg_align = True
arg_normalization = "base" # Default is base. Options: base, raw, Facenet, Facenet2018, VGGFace, VGGFace2, ArcFace
arg_distance_metric = "cosine" # Options: 'cosine', 'euclidean', 'euclidean_l2'.
arg_threshold = 0.50


identification_result = face_identification(resp_objs=detect_result, df=db_dataframe, face_recognitor=face_recognitor, enforce_detection=arg_enforce_detection, align=arg_align, normalization=arg_normalization, distance_metric=arg_distance_metric, threshold=arg_threshold)

print(file_path)
print(identification_result)