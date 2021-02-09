import os
import tensorflow as tf

class CommonConfig :
    PROJECT_PATH = "C:/Users/rlfalsgh95/source/repos/bondinet"  # 프로젝트의 경로
    ORIGINAL_IMG_DIR_PATH = "C:/Users/rlfalsgh95/Desktop"   # 원본 SMDB가 저장된 위치

    IMG_ROOT_DIR_NAME = "SMDB"  # 이미지가 저장된 ROOT 디렉터리명
    SMART_PHONE_IMG_DIR_NAME = "smartphone_photo"   # 스마트폰 이미지가 저장된 디렉터리명
    CAMERA_IMG_DIR_NAME = "camera_photo"    # 카메라 이미지가 저장된 디렉터리명

    PHONE_IMG_DIR_UNZIP_PATH = os.path.join(PROJECT_PATH, IMG_ROOT_DIR_NAME, SMART_PHONE_IMG_DIR_NAME)  # 원본 스마트폰 이미지를 압축 해제할 디렉토리 경로
    CAMERA_IMG_DIR_UNZIP_PATH = os.path.join(PROJECT_PATH, IMG_ROOT_DIR_NAME, CAMERA_IMG_DIR_NAME) # 원본 DSLR 이미지를 압축 해제할 디렉토리 경로

    ORIGINAL_PHONE_IMG_DIR_PATH = os.path.join(ORIGINAL_IMG_DIR_PATH, IMG_ROOT_DIR_NAME, SMART_PHONE_IMG_DIR_NAME)  # 원본 스마트폰 이미지의 디렉토리 경로
    ORIGINAL_CAMERA_IMG_DIR_PATH = os.path.join(ORIGINAL_IMG_DIR_PATH, IMG_ROOT_DIR_NAME, CAMERA_IMG_DIR_NAME)  # 원본 DSLR 이미지의 디렉토리 경로

    IMG_DATA_PICKLE_NAME = "img_data.pickle"    # 이미지 데이터와 레이블을 저장할 pickle 파일명


class PreprocessingConfig :
    # load_img_data() 파라미터
    MAX_IMG_PER_CLASS = 100 # 모델당 패치 1000장정도, 29개 모델 (patch 사이즈/개수, center crop size는 알아서)
    RESUME = True
    SAVE = True
    IMG_WIDTH = 64
    IMG_HEIGHT = 64

class TrainConfig :
    CLASS_NUM = 29  # Galaxy Note9가 있기 때문에 Galaxy Note9 SM-N960N 디렉터리는 제거하고, Galaxy A8와 Galaxy A8(2018)은 서로 다른 클래스로 취급
    BATCH_SIZE = 128
    EPOCH = 100
    MOMEMTUM = 0.9
    WEIGHT_DECAY = 0.00075
    BASE_LR = 0.01 # TODO : learning rate (learning rate is initialized to 0.015 and halves every 10 epochs.)

    BATCH_RANDOM_SEED = 42
    MODEL_SAVE_INTERVAL = 100   # Save Model per MODEL_SAVE_INTERVAL epoch
    LOG_WRITE_INTERVAL = 10

    MODEL_FILE_NAME = "cnn_model"
    LOG_DIR_NAME = "logs"

    # conv common parameter
    CONV_PADDING = "valid"
    CONV_ACTIVATION = None
    CONV_KERNEL_INITIALIZER = "glorot_uniform"
    CONV_BIAS_INITIALIZER = "constant"

    # l1 conv parameter
    L1_CONV_FILTERS = 32
    L1_CONV_KERNEL_SIZE = (4, 4)

    # l2 conv parameter
    L2_CONV_FILTERS = 48
    L2_CONV_KERNEL_SIZE = (5, 5)

    # l3 conv parameter
    L3_CONV_FILTERS = 64
    L3_CONV_KERNEL_SIZE = (5, 5)

    # l4 conv parameter
    L4_CONV_FILTERS = 128
    L4_CONV_KERNEL_SIZE = (5, 5)

    # max_pooling parameter
    MAX_POOLING_POOL_SIZE = [2, 2]
    MAX_POOLING_STRIDES = [2, 2]
    MAX_POLLING_PADDING = "same"    # TODO : max_pooling padding 수정

    # ip layer parameter
    UNITS = 128
    ACTIVATION_FUNC = tf.nn.relu
