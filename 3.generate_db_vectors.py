import os
from xperty.lib_db import generate_db_vectors
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

generate_db_vectors(
    db_path='../tmp/db/',
    model_name='ArcFace',
    detector_backend='yolov8',
)