import data_processing
from config import CommonConfig as cc
from config import TrainConfig as tc
from config import PreprocessingConfig as pc
from bondinet import Bondinet
import os

# TODO : 함수 주석 달기 및 모듈화
# TODO : Galaxy S6_3552.jpg 이미지 깨진거 보고
# TODO : Dresden.csv와 비교했을 때, 73장의 이미지가 부족함.

# 창희님 네트워크
# 원본 이미지를 겹치지 않게 256 * 256 패치 추출, 모두 사용
# 각 패치에 대해 분류, 패치들의 평균 정확도를 사용

# 실험
# 이미지 저장 시에는 .npy 또는 tensorflow의 record 사용
# 학습 시, keras의 generator 또는 tensorflow의 dictionary 사용
# 첫 에폭 로스가 대략 log2(클래스 개수)정도면 나쁜 네트워크는 아님
# 스마트폰, DSLR 나누어서 훈련
# train, test, val은 8 : 2, 8 : 2 사용 (모델 비율에 따라서 분할)
# HEIC 이미지는 사용하지 않음.
# 두 개 네트워크 전처리는 맞추지 않음
# TODO : A8, A8(2018)은 일단 다른 클래스로 둠. 나중에 다시 정해봐야 봐야할 듯 
# TODO : Galaxy Note9 SM-N960N는 제외한 후 실험 (Note 9과 합쳐서 실험 - 플래그쉽 모델은 성능이 달라지지 않아서 합쳐서 사용하고, A8은 보급형이라 년도마다 성능이 다름 요것도 한번 확인해봐야할 듯)
# TODO : bondinet 코드 사용을 제한하고 있음, patch_extractor은 고려해봐야할 듯

# TODO : Dresden 모델명 추출 방법 생각해보기
# TODO : dresden 64 * 64 * 3, 256 * 256 * 3 크기의 패치로 나누기
# TODO : dresden 데이터셋은 장면 분할 사용
# TODO : Train/Val/Test set 분할 
# TODO : config 파일의 파라미터는 되도록 호출부에서 전달하도록 수정
# TODO : bondinet 생성자에 learning scheduler 사용 여부 추가 하기
# TODO : 사용되지 않는 파라미터가 전달된 경우, error 발생하도록 수정 (ex. LEARNING_SCHEDULER를 지정하지 않았는데, DECAY_RATE를 지정한 경우)
# TODO : optimizer의 종류도 모델 디렉토리명에 포함되도록 수정
# TODO : 저장한 모델을 불러올 수 있도록 수정
# TODO : confusion matrix 그리기
# TODO : sgd + momentum + learning rate scheduler 다시 실험해보기
# TODO : loss 계속 튕기면 데이터 한번 확인해보기
# TODO : 네트워크 학습 속도 개선 (gpu volatile util - https://newsight.tistory.com/255, multi processing)
# TODO : 이미지들이 공통적으로 갖는 SCENE을 기준으로 TRAIN과 TEST를 분할하도록 해보는 것도 생각해봐야할 듯
# TODO : tensorboard 히스토그램은 gradient, variable처럼 같이 나오도록 하고, scalar는 min이면, min, std면 std 그룹지어서 보여주기
# TODO : 두개 미만의 장치도 사용하는 경우 instance split은 사용 x
# TODO : 실행 순서대로 py 파일에 숫자 붙이기

# TODO : 다음 세미나 때까지 실험
# 1. non overlapping/overlapping, 패치 개수 달리해서 실험 (overlapping의 경우 패치 개수 많이 사용, non overlapping/overlapping는 네트워크 하나에 대해 실험)
# 2. quality function 적용 결과 0 ~ 1 사이의 값이 나오는지, flat/saturated pixel/ standard deviation 낮은 이미지 score 낮은지 확인
# 3. 이미지 전체의 퀄리티 점수가 낮은 이미지의 분류 성능은?
# 4. 본디넷 구현 세미나할 때, ovo/ova 좀 더 자세하게 설명
# 5. 장치 개수 상관없이 모든 모델을 사용한 거 / 모델 당 장치 개수 2개 이상인 모델만 사용한 거 실험

# TODO : 학습 시 변경사항 체크리스트
# 1. optimizer, learning rate schedule 사용 여부(사용하는 경우 BASE_LR 수정 0.001 -> 0.01)
# 2. MODEL_SAVE_INTERVAL
# 3. CLASS_NUM
# 4. TEST, VALIDATION, TEST split 방법 (장면 분할 or 무작위 분할)
# 5. image dataset 종류
# 6. 나머지 하이퍼 파라미터
# 7. 기존 저장된 모델 파일이 있는지, 덮어씌워져도 되는지
# 8. 사용하지 않는 파라미터에 값이 지정됐는지 확인


# 질문
# TODO : dense layer의 bias도 사용하지 않는건가?
# TODO : Dresden 디렉터리의 ver 폴더는 무엇인지 
# TODO : motive란?

# TODO : machine learning에 gpu가 사용되는 이유
# TODO : gpu의 용량은 무엇을 의미?
# TODO : 이미지를 numpy로 변환하지 않고, 이미지 그대로를 network에 입력하는 방법은?
# TODO : compute_gradients가 반환하는 variable과 weight는 뭐가 다른지.
# TODO : gradient와 variable의 histogram을 그릴 때, l2_norm은 왜 적용했는지
# TODO : tensorboard의 Distribution 탭은 무엇을 의미하는지

if __name__ == "__main__" : 
    # image unzip
    # data_processing.unzip(source_path = cc.ORIGINAL_IMG_ZIP_PATH, dest_path = "C:/Users/rlfalsgh95/source/repos/bondinet/SMDB")
    
    load_img_option = {"max_img_per_class" : pc.MAX_IMG_PER_CLASS, "resume" : pc.RESUME_LOAD}

    if pc.CENTER_CROP : # CENTER_CROP 파리미터가 True인 경우
        load_img_option["center_crop"] = {"crop_width" : pc.IMG_CROP_WIDTH, "crop_height" : pc.IMG_CROP_HEIGHT} # crop 관련 파라미터 추가
    elif pc.EXTRACT_PATCH : # EXTRACT_PATCH 파라미터가 True인 경우 patch 추출 파라미터 추가
        load_img_option["patch_option"] = {'dim': pc.PATCH_DIM,
        'offset': pc.PATCH_OFFSET,
        'stride': pc.PATCH_STRIDE,
        'rand': pc.PATCH_RAND,
        'function': pc.PATCH_HANDLER,
        'threshold': pc.PATCH_THRESHOLD,
        'num': pc.N_MAX_PATCH,
        }

    img_dir_path_list = [cc.PHONE_IMG_DIR_UNZIP_PATH] # 스마트폰 이미지만 사용
    img_data, label, img_to_array_info = data_processing.img_to_array(img_dir_path_list, cc.PICKLE_DIR_PATH, option = load_img_option)  # 이미지 데이터와 레이블 불러옴

    print("img_to_array error_cnt : ", img_to_array_info["error_cnt"], "\n", img_to_array_info["error_file_list"])  # image를 array로 변환할 때 에러가 발생한 파일의 개수와 파일명 출력
    
    model_dir_path = os.path.join(img_to_array_info["pickle_dir_path"], cc.MODEL_DIR_NAME, f"{tc.TRAIN_RATE}_{tc.VALIDATION_RATE}_{tc.TEST_RATE}_{tc.BATCH_SIZE}_{tc.EPOCH}_{tc.MOMENTUM}_{tc.WEIGHT_DECAY}_{tc.BASE_LR}_{tc.LR_DECAY_INTERVAL}_{tc.LR_DECAY_RATE}") # 모델을 저장할 디렉토리 path
    
    encoder, oneHotLabel = data_processing.encode_label(label, onehot = True)    # string label을 one hot vector로 변환


    bondinet = Bondinet(n_classes = tc.N_CLASSES, 
                        base_lr=tc.BASE_LR, weight_decay = tc.WEIGHT_DECAY, 
                        momentum = tc.MOMENTUM, lr_decay_interval = tc.LR_DECAY_INTERVAL, lr_decay_rate = tc.LR_DECAY_RATE)   # bondinet 모델 생성

    bondinet.train(X_data = img_data, y_data = oneHotLabel , 
                   n_epoch = tc.EPOCH, batch_size = tc.BATCH_SIZE, 
                   model_dir_path = model_dir_path, batch_random_seed = tc.BATCH_RANDOM_SEED,
                   model_save_interval = tc.MODEL_SAVE_INTERVAL, log_write_interval = tc.LOG_WRITE_INTERVAL)   # CNN 모델 학습

# 완료
# dresden, SMDB에 따라 이미지 데이터 분할되도록 수정 (ex. data/original/SMDB, data/dresden/original/Dresden)
# SMDB 사용하지 않는 모델, 확장자 제거
# 이벤트 파일 저장하도록 해서, tensorboard로 그래프 볼 수 있도록 수정
# dresden 이미지 데이터셋 프로젝트 path로 옮기기

