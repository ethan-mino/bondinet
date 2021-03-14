import os
import patch_extractor

class CommonConfig :
    ZIP_FILE_EXT = ".zip"
    DESKTOP_DIR_PATH = "C:/Users/rlfalsgh95/Desktop"    # 데스크탑
    PROJECT_PATH = "C:/Users/rlfalsgh95/source/repos/bondinet"  # 프로젝트의 경로
    DATA_DIR_PATH = os.path.join(PROJECT_PATH, "data")  # array로 변환된 이미지 데이터, pickle 파일, 모델 학습 파일, 이벤트 파일 등의 데이터를 저장할 디렉토리

    DRESDEN_NPY_FILE_PATH = DRESDEN_CSV_FILE_PATH = os.path.join(PROJECT_PATH, "dresden.npy")
    DRESDEN_CSV_FILE_PATH = os.path.join(PROJECT_PATH, "dresden.csv")

    IMG_DIR_NAME = "SMDB"  # 이미지가 저장된 디렉터리명
    UNZIPED_ORIGINAL_IMG_DIR_NAME = "original"  # 압축 해제가 완료된 원본 이미지
    MODEL_DIR_NAME = "model"
    PICKLE_DIR_NAME = "pickle,model"
    SMART_PHONE_IMG_DIR_NAME = "smartphone_photo"   # 스마트폰 이미지가 저장된 디렉터리명
    CAMERA_IMG_DIR_NAME = "camera_photo"    # 카메라 이미지가 저장된 디렉터리명

    ORIGINAL_IMG_ZIP_PATH = os.path.join(DESKTOP_DIR_PATH, IMG_DIR_NAME + ZIP_FILE_EXT) # 압축해제 되지 않은 원본 이미지가 저장된 zip file path

    PHONE_IMG_DIR_UNZIP_PATH = os.path.join(DATA_DIR_PATH, UNZIPED_ORIGINAL_IMG_DIR_NAME, IMG_DIR_NAME, SMART_PHONE_IMG_DIR_NAME)  # 원본 스마트폰 이미지를 압축 해제할 디렉토리 경로
    CAMERA_IMG_DIR_UNZIP_PATH = os.path.join(DATA_DIR_PATH, UNZIPED_ORIGINAL_IMG_DIR_NAME, IMG_DIR_NAME, CAMERA_IMG_DIR_NAME) # 원본 DSLR 이미지를 압축 해제할 디렉토리 경로

    UNZIPED_ORIGINAL_IMG_DIR_PATH = os.path.join(DATA_DIR_PATH, UNZIPED_ORIGINAL_IMG_DIR_NAME)
    PICKLE_DIR_PATH = os.path.join(DATA_DIR_PATH, PICKLE_DIR_NAME, IMG_DIR_NAME)
    
class PreprocessingConfig : 
    DATA_SPLIT_SEED = 95    # train, val, test dataset을 분할하는데 사용되는 seed
    TRAIN_RATE = 0.64
    VALIDATION_RATE = 0.16
    TEST_RATE = 0.2
    
    # Dresden dataset parameter
    SINGLE_DEVICE = False   # 하나의 장치로만 구성된 모델을 포함하는지 나타내는 parameter

    # load_img_data() 파라미터
    RESUME_LOAD = True

    CENTER_CROP = False
    IMG_CROP_WIDTH = None
    IMG_CROP_HEIGHT = None

    # patch config
    EXTRACT_PATCH = True
    PATCH_DIM = (64,64)  # the dimensions of the patches (rows,cols).
    PATCH_OFFSET = 0  # the offsets of each axis starting from top left corner (rows,cols).
    PATCH_STRIDE = 64  # the stride of each axis starting from top left corner (rows,cols).
    PATCH_RAND = False    # rand patches. Must not be set together with function
    PATCH_HANDLER = patch_extractor.mid_intensity_high_texture   # patch quality function handler. Must not be set together with rand
    PATCH_THRESHOLD = .0   # minimum quality threshold
    N_MAX_PATCH = 32    # maximum number of patches
   

class TrainConfig :
    GPU_INDEX = "1"
    RESUME_TRAIN = True
    BATCH_RANDOM_SEED = 42

    N_CLASSES = 29  # Galaxy Note9가 있기 때문에 Galaxy Note9 SM-N960N 디렉터리는 제거하고, Galaxy A8와 Galaxy A8(2018)은 서로 다른 클래스로 취급
    
    BATCH_SIZE = 128
    EPOCH = 100
    
    MOMENTUM = 0.9
    WEIGHT_DECAY = 0.00075

    # learning rate is initialized to 0.015 and halves every 10 epochs.
    OPTIMIZER = "Adam"
    BASE_LR = 0.001 # Adam : 0.001, SGD : 0.01
    LR_SCHEDULER = False
    LR_DECAY_INTERVAL = None # Adam : None, SGD : 10
    LR_DECAY_RATE = None # Adam : None, SGD : 0.5

    MODEL_SAVE_INTERVAL = 1   # Save Model per MODEL_SAVE_INTERVAL epoch 
    LOG_WRITE_INTERVAL = 10 # write log per LOG_WRITE_INTERVAL iteration 
