import numpy as np
from typing import Any, Dict, List, Union
import pickle
import pandas as pd

from deepface.commons import image_utils
from deepface.modules import detection, verification, modeling
from deepface.models.FacialRecognition import FacialRecognition
from deepface.modules import modeling, detection, preprocessing



def load_db_vectors(pkl_path):
    with open(pkl_path, "rb") as f:
        db_dataframe = pd.DataFrame(pickle.load(f))
    return db_dataframe


def recognitor_build(model_str):
    face_recognitor: FacialRecognition = modeling.build_model(model_str)
    return face_recognitor



def represent(
    face_recognitor,
    img_path: Union[str, np.ndarray],
    enforce_detection: bool = True,
    detector_backend: str = "opencv",
    align: bool = True,
    expand_percentage: int = 0,
    normalization: str = "base",
) -> List[Dict[str, Any]]:
    """
    Represent facial images as multi-dimensional vector embeddings.

    Args:
        img_path (str or np.ndarray): The exact path to the image, a numpy array in BGR format,
            or a base64 encoded image. If the source image contains multiple faces, the result will
            include information for each detected face.

        enforce_detection (boolean): If no face is detected in an image, raise an exception.
            Default is True. Set to False to avoid the exception for low-resolution images.

        detector_backend (string): face detector backend. Options: 'opencv', 'retinaface',
            'mtcnn', 'ssd', 'dlib', 'mediapipe', 'yolov8', 'centerface' or 'skip'.

        align (boolean): Perform alignment based on the eye positions.

        expand_percentage (int): expand detected facial area with a percentage (default is 0).

        normalization (string): Normalize the input image before feeding it to the model.
            Default is base. Options: base, raw, Facenet, Facenet2018, VGGFace, VGGFace2, ArcFace

    Returns:
        results (List[Dict[str, Any]]): A list of dictionaries, each containing the
            following fields:

        - embedding (List[float]): Multidimensional vector representing facial features.
            The number of dimensions varies based on the reference model
            (e.g., FaceNet returns 128 dimensions, VGG-Face returns 4096 dimensions).
        - facial_area (dict): Detected facial area by face detection in dictionary format.
            Contains 'x' and 'y' as the left-corner point, and 'w' and 'h'
            as the width and height. If `detector_backend` is set to 'skip', it represents
            the full image area and is nonsensical.
        - face_confidence (float): Confidence score of face detection. If `detector_backend` is set
            to 'skip', the confidence will be 0 and is nonsensical.
    """
    resp_objs = []

    # ---------------------------------
    # we have run pre-process in verification. so, this can be skipped if it is coming from verify.
    target_size = face_recognitor.input_shape
    if detector_backend != "skip":
        img_objs = detection.extract_faces(
            img_path=img_path,
            detector_backend=detector_backend,
            grayscale=False,
            enforce_detection=enforce_detection,
            align=align,
            expand_percentage=expand_percentage,
        )
    else:  # skip
        # Try load. If load error, will raise exception internal
        img, _ = image_utils.load_image(img_path)

        if len(img.shape) != 3:
            raise ValueError(f"Input img must be 3 dimensional but it is {img.shape}")

        # make dummy region and confidence to keep compatibility with `extract_faces`
        img_objs = [
            {
                "face": img,
                "facial_area": {"x": 0, "y": 0, "w": img.shape[0], "h": img.shape[1]},
                "confidence": 0,
            }
        ]
    # ---------------------------------

    for img_obj in img_objs:
        img = img_obj["face"]

        # rgb to bgr
        img = img[:, :, ::-1]

        region = img_obj["facial_area"]
        confidence = img_obj["confidence"]

        # resize to expected shape of ml model
        img = preprocessing.resize_image(
            img=img,
            # thanks to DeepId (!)
            target_size=(target_size[1], target_size[0]),
        )

        # custom normalization
        img = preprocessing.normalize_input(img=img, normalization=normalization)

        embedding = face_recognitor.forward(img)

        resp_obj = {}
        resp_obj["embedding"] = embedding
        resp_obj["facial_area"] = region
        resp_obj["face_confidence"] = confidence
        resp_objs.append(resp_obj)

    return resp_objs

def face_identification(resp_objs, df, face_recognitor, enforce_detection, align, normalization, distance_metric, threshold):
    resp_obj = []
    for source_obj in resp_objs:
        source_img = source_obj["face"]
        source_region = source_obj["facial_area"]
        target_embedding_obj = represent(
            face_recognitor=face_recognitor,
            img_path=source_img,
            enforce_detection=enforce_detection,
            detector_backend="skip",
            align=align,
            normalization=normalization,)
        target_representation = target_embedding_obj[0]["embedding"]

        result_df = df.copy()  # df will be filtered in each img
        result_df["source_x"] = source_region["x"]
        result_df["source_y"] = source_region["y"]
        result_df["source_w"] = source_region["w"]
        result_df["source_h"] = source_region["h"]

        distances = []
        for _, instance in df.iterrows():
            source_representation = instance["embedding"]
            if source_representation is None:
                distances.append(float("inf"))  # no representation for this image
                continue

            target_dims = len(list(target_representation))
            source_dims = len(list(source_representation))
            if target_dims != source_dims:
                raise ValueError(
                    "Source and target embeddings must have same dimensions but "
                    + f"{target_dims}:{source_dims}. Model structure may change"
                    + " after pickle created. Delete the {file_name} and re-run."
                )

            distance = verification.find_distance(
                source_representation, target_representation, distance_metric
            )

            distances.append(distance)

        target_threshold = threshold or verification.find_threshold('yolov8', distance_metric)

        result_df["threshold"] = target_threshold
        result_df["distance"] = distances

        result_df = result_df.drop(columns=["embedding"])
        # pylint: disable=unsubscriptable-object
        result_df = result_df[result_df["distance"] <= target_threshold]
        result_df = result_df.sort_values(by=["distance"], ascending=True).reset_index(drop=True)

        resp_obj.append(result_df)
    return resp_obj



# # # #### forward code ####
# import glob, os
# from xperty import lib_det

# arg_enforce_detection = True
# arg_expand_percentage = 0
# arg_align = True
# arg_normalization = "base" # Default is base. Options: base, raw, Facenet, Facenet2018, VGGFace, VGGFace2, ArcFace
# arg_distance_metric = "cosine" # Options: 'cosine', 'euclidean', 'euclidean_l2'.
# arg_threshold = 0.68

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# # Load the representations from the pickle file
# db_dataframe = load_db_vectors('../tmp/db/ds_model_arcface_detector_yolov8_aligned_normalization_base_expand_0_80.pkl')

# models = ["VGG-Face", "Facenet", "Facenet512", "OpenFace", "DeepFace", 
#           "DeepID", "ArcFace", "Dlib", "SFace", "GhostFaceNet",]
# model = models[6]

# face_recognitor = recognitor_build(model_str=model)
# face_detector = lib_det.detector_build(backend_str='yolov8')


# file_path_list = glob.glob('../20240603/*/*')
# file_path = file_path_list[0]
# img, img_name = image_utils.load_image(file_path)


# detect_result = lib_det.face_detect(img, img_name, detector=face_detector, align=arg_align, expand_percentage=arg_expand_percentage, enforce_detection=arg_enforce_detection)
# identification_result = face_identification(resp_objs=detect_result, df=db_dataframe, face_recognitor=face_recognitor, enforce_detection=arg_enforce_detection, align=arg_align, normalization=arg_normalization, distance_metric=arg_distance_metric, threshold=arg_threshold)

# print(file_path)
# print(identification_result)
# # # ########################