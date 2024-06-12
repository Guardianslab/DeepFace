import numpy as np
import cv2

from deepface.models.Detector import DetectedFace, FacialAreaRegion
from deepface.detectors import DetectorWrapper
from deepface.modules import detection




def detector_build(backend_str):
    face_detector: DetectorWrapper.Detector = DetectorWrapper.build_model(backend_str)
    return face_detector
    

def face_detect(img, img_name, detector, align, expand_percentage, enforce_detection):
    resp_objs = []

    base_region = FacialAreaRegion(x=0, y=0, w=img.shape[1], h=img.shape[0], confidence=0)

    height, width, _ = img.shape

    # If faces are close to the upper boundary, alignment move them outside
    # Add a black border around an image to avoid this.
    height_border = int(0.5 * height)
    width_border = int(0.5 * width)
    if align is True:
        img = cv2.copyMakeBorder(
            img,
            height_border,
            height_border,
            width_border,
            width_border,
            cv2.BORDER_CONSTANT,
            value=[0, 0, 0],  # Color of the border (black)
        )

    # find facial areas of given image
    facial_areas = detector.detect_faces(img)

    face_objs = []
    for facial_area in facial_areas:
        x = facial_area.x
        y = facial_area.y
        w = facial_area.w
        h = facial_area.h
        left_eye = facial_area.left_eye
        right_eye = facial_area.right_eye
        confidence = facial_area.confidence

        if expand_percentage > 0:
            # Expand the facial region height and width by the provided percentage
            # ensuring that the expanded region stays within img.shape limits
            expanded_w = w + int(w * expand_percentage / 100)
            expanded_h = h + int(h * expand_percentage / 100)

            x = max(0, x - int((expanded_w - w) / 2))
            y = max(0, y - int((expanded_h - h) / 2))
            w = min(img.shape[1] - x, expanded_w)
            h = min(img.shape[0] - y, expanded_h)

        # extract detected face unaligned
        detected_face = img[int(y) : int(y + h), int(x) : int(x + w)]

        # align original image, then find projection of detected face area after alignment
        if align is True:  # and left_eye is not None and right_eye is not None:
            aligned_img, angle = detection.align_face(
                img=img, left_eye=left_eye, right_eye=right_eye
            )

            rotated_x1, rotated_y1, rotated_x2, rotated_y2 = DetectorWrapper.rotate_facial_area(
                facial_area=(x, y, x + w, y + h), angle=angle, size=(img.shape[0], img.shape[1])
            )
            detected_face = aligned_img[
                int(rotated_y1) : int(rotated_y2), int(rotated_x1) : int(rotated_x2)
            ]

            # restore x, y, le and re before border added
            x = x - width_border
            y = y - height_border
            # w and h will not change
            if left_eye is not None:
                left_eye = (left_eye[0] - width_border, left_eye[1] - height_border)
            if right_eye is not None:
                right_eye = (right_eye[0] - width_border, right_eye[1] - height_border)

        result = DetectedFace(
            img=detected_face,
            facial_area=FacialAreaRegion(
                x=x, y=y, h=h, w=w, confidence=confidence, left_eye=left_eye, right_eye=right_eye
            ),
            confidence=confidence,
        )
        face_objs.append(result)

    # in case of no face found
    if len(face_objs) == 0 and enforce_detection is True:
        if img_name is not None:
            raise ValueError(
                f"Face could not be detected in {img_name}."
                "Please confirm that the picture is a face photo "
                "or consider to set enforce_detection param to False."
            )
        else:
            raise ValueError(
                "Face could not be detected. Please confirm that the picture is a face photo "
                "or consider to set enforce_detection param to False."
            )
        
    if len(face_objs) == 0 and enforce_detection is False:
        face_objs = [DetectedFace(img=img, facial_area=base_region, confidence=0)]

    for face_obj in face_objs:
        current_img = face_obj.img
        current_region = face_obj.facial_area

        if current_img.shape[0] == 0 or current_img.shape[1] == 0:
            continue

        current_img = current_img / 255  # normalize input in [0, 1]

        resp_objs.append(
            {
                "face": current_img[:, :, ::-1],
                "facial_area": {
                    "x": int(current_region.x),
                    "y": int(current_region.y),
                    "w": int(current_region.w),
                    "h": int(current_region.h),
                    "left_eye": current_region.left_eye,
                    "right_eye": current_region.right_eye,
                },
                "confidence": round(current_region.confidence, 2),
            }
        )
    
    if len(resp_objs) == 0 and enforce_detection == True:
        raise ValueError(
            f"Exception while extracting faces from {img_name}."
            "Consider to set enforce_detection arg to False."
        )
    return resp_objs









# # #### forward code ####
# import glob, os
# from deepface.commons import image_utils


# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# arg_enforce_detection = True
# arg_expand_percentage = 0
# arg_align = True
# arg_threshold = 0.68

# backends = ['opencv', 'ssd', 'dlib', 'mtcnn', 'fastmtcnn',
#   'retinaface', 'mediapipe', 'yolov8', 'yunet', 'centerface',]
# backend = backends[7]

# face_detector = detector_build(backend_str=backend)

# file_path_list = glob.glob('../20240603/*/*')
# file_path = file_path_list[0]
# img, img_name = image_utils.load_image(file_path)

# detect_result = face_detect(img, img_name, detector=face_detector, align=arg_align, expand_percentage=arg_expand_percentage, enforce_detection=arg_enforce_detection)
# # ########################