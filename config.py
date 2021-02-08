import os

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
    MAX_IMG_PER_CLASS = 1
    RESUME = True
    SAVE = True
    IMG_WIDTH = 64
    IMG_HEIGHT = 64

class TrainConfig : 
    CLASS_NUM = 42