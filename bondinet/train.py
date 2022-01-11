from data_processing import encode_label, img_to_array
from config import CommonConfig as cc
from config import TrainConfig as tc
from config import PreprocessingConfig as pc
from bondinet import Bondinet
from scene_split import scene_split 
from itertools import groupby

import os
from sklearn.model_selection import train_test_split

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
# Dresden.csv와 비교했을 때, 73장의 이미지가 부족함. (Dresden.npy와는 72장)
# dresden.cvs에는 shot index가 43583인 이미지 정보가 있지만, dresden.npy에서는 제외되어있음. (실제 소유중인 Dresden 데이터셋에도 없음)
# TODO : A8, A8(2018)은 일단 다른 클래스로 둠. 나중에 다시 정해봐야 봐야할 듯 
# TODO : Galaxy Note9 SM-N960N는 제외한 후 실험 (Note 9과 합쳐서 실험 - 플래그쉽 모델은 성능이 달라지지 않아서 합쳐서 사용하고, A8은 보급형이라 년도마다 성능이 다름 요것도 한번 확인해봐야할 듯)
# TODO : bondinet 코드 사용을 제한하고 있음, patch_extractor은 고려해봐야할 듯

# TODO : constant initializer의 초기 값 확인
# TODO : 이미지 스케일링 방식 변경
# TODO : 네트워크가 이미지를 입력받으면 standardization이나, 논문처럼 scale을 변환하도록 수정
# TODO : Galaxy S6_3552.jpg 이미지 깨진거 보고
# TODO : sgd + momentum + learning rate scheduler 다시 실험해보기
# TODO : loss 계속 튕기면 데이터 한번 확인해보기
# TODO : 네트워크 학습 속도 개선 (gpu volatile util - https://newsight.tistory.com/255, multi processing)
# TODO : 두개 미만의 장치도 사용하는 경우 instance split은 사용 x
# TODO : Test 결과 confusion matrix 그리기
# TODO : DIDB/SMDB 256 * 256 * 3 크기의 패치로 나누기
# TODO : scene split시에 이미지들이 공통적으로 갖는 SCENE을 기준으로 TRAIN과 TEST를 분할하도록 해보는 것도 생각해봐야할 듯
# TODO : f"{}"형식 -> "{}".format()
# TODO : config 파일의 파라미터는 되도록 호출부에서 전달하도록 수정
# TODO : 실행 순서대로 py 파일에 숫자 붙이기
# TODO : 함수 주석 달기 및 모듈화

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
    
    load_img_option = {"resume" : pc.RESUME_LOAD}

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
    
    pickle_dir_path = os.path.join(cc.PICKLE_DIR_PATH, f"{pc.TRAIN_RATE}_{pc.VALIDATION_RATE}_{pc.TEST_RATE}_{pc.SINGLE_DEVICE}_{pc.CENTER_CROP}_{pc.IMG_CROP_WIDTH}_{pc.IMG_CROP_HEIGHT}_{pc.EXTRACT_PATCH}_{pc.PATCH_DIM}_{pc.PATCH_OFFSET}_{pc.PATCH_STRIDE}_{pc.PATCH_RAND}_{pc.PATCH_THRESHOLD}_{pc.N_MAX_PATCH}")

    if cc.IMG_DIR_NAME == "DIDB" :  # Dreden 데이터 셋을 사용하는 경우
        data_set = scene_split(single_device = pc.SINGLE_DEVICE, seed = pc.DATA_SPLIT_SEED)   # dresden dataset scene split
        
        origin_X_train, origin_y_train = data_set["X_train"], data_set["y_train"]
        origin_X_val, origin_y_val = data_set["X_val"], data_set["y_val"]
        origin_X_test, origin_y_test = data_set["X_test"], data_set["y_test"]
    elif cc.IMG_DIR_NAME == "SMDB" : # 실험 데이터 셋을 사용하는 경우
        model_name_list = os.listdir(cc.PHONE_IMG_DIR_UNZIP_PATH) # 스마트폰 이미지만 사용  # TODO : DSLR로 변경하는 경우 기존 데이터 옮기기
        X, y = [], [] 
        for model_name in model_name_list : 
            model_dir_path = os.path.join(cc.PHONE_IMG_DIR_UNZIP_PATH, model_name)  # 해당 카메라 모델의 이미지를 저장하는 dir의 path
            for path, dirs, files in os.walk(model_dir_path) :  # 해당 디렉토리 하위의 모든 파일 path와 모델명을 append
                X += [os.path.join(path, file) for file in files]
                y += [model_name for i in range(len(files))]

        origin_X_train, origin_X_test, origin_y_train, origin_y_test = train_test_split(X, y, test_size = 1 - pc.TRAIN_RATE, random_state = pc.DATA_SPLIT_SEED) # https://datascience.stackexchange.com/questions/15135/train-test-validation-set-splitting-in-sklearn
        origin_X_val, origin_X_test, origin_y_val, origin_y_test= train_test_split(origin_X_test, origin_y_test, test_size = pc.TEST_RATE / (pc.TEST_RATE + pc.VALIDATION_RATE) , random_state = pc.DATA_SPLIT_SEED)
        assert len(X), len(origin_X_train) + len(origin_X_val) + len(origin_X_test) # train, test, val에 포함되지 않은 이미지가 있는지 확인(dresden dataset은 scene_split()에서 확인)
    
    X_train, y_train = img_to_array(origin_X_train, origin_y_train, os.path.join(pickle_dir_path, "Train"), option = load_img_option)  # Train_set 이미지 데이터와 레이블 불러옴
    X_val, y_val = img_to_array(origin_X_val, origin_y_val, os.path.join(pickle_dir_path, "Val"), option = load_img_option)  # Val_set 이미지 데이터와 레이블 불러옴
    X_test, y_test = img_to_array(origin_X_test, origin_y_test, os.path.join(pickle_dir_path, "Test"), option = load_img_option)  # Test_set 이미지 데이터와 레이블 불러옴
    
    """
    import numpy as np
    print("train")
    X_data = sorted(np.c_[range(len(origin_X_train)), origin_y_train], key = lambda x : x[1])  # 모델을 기준으로 정렬 (https://hulk89.github.io/python/2016/11/25/groupby/)

    for key, items in groupby(X_data, lambda x : x[1]) :    # 모델별 정보로 변환
        print("model_name : ", key, "# : ", len([item for item in list(items)]))    # 모델당 이미지 개수 출력


    print("test")
    X_data = sorted(np.c_[range(len(origin_X_test)), origin_y_test], key = lambda x : x[1])  # 모델을 기준으로 정렬 (https://hulk89.github.io/python/2016/11/25/groupby/)

    for key, items in groupby(X_data, lambda x : x[1]) :    # 모델별 정보로 변환
        print("model_name : ", key, "# : ", len([item for item in list(items)]))    # 모델당 이미지 개수 출력

    print("val")
    X_data = sorted(np.c_[range(len(origin_X_val)), origin_y_val], key = lambda x : x[1])  # 모델을 기준으로 정렬 (https://hulk89.github.io/python/2016/11/25/groupby/)

    for key, items in groupby(X_data, lambda x : x[1]) :    # 모델별 정보로 변환
        print("model_name : ", key, "# : ", len([item for item in list(items)]))    # 모델당 이미지 개수 출력
    """

    assert len(origin_X_train) * pc.N_MAX_PATCH if pc.EXTRACT_PATCH else len(origin_X_train), len(X_train) # 이미지 데이터 변환 전후의 이미지 개수가 다르면 error 발생 (patch를 추출하는 경우 이미지 개수 * patch 최대 개수)
    assert len(origin_X_val) * pc.N_MAX_PATCH if pc.EXTRACT_PATCH else len(origin_X_val), len(X_val)
    # TODO : assert len(origin_X_test) * pc.N_MAX_PATCH if pc.EXTRACT_PATCH else len(origin_X_test), len(X_test)
            
    model_dir_path = os.path.join(pickle_dir_path, cc.MODEL_DIR_NAME, f"{tc.N_CLASSES}_{tc.BATCH_SIZE}_{tc.EPOCH}_{tc.MOMENTUM}_{tc.WEIGHT_DECAY}_{tc.OPTIMIZER}_{tc.BASE_LR}_{tc.LR_SCHEDULER}_{tc.LR_DECAY_INTERVAL}_{tc.LR_DECAY_RATE}") # 모델을 저장할 디렉토리 path
    
    encoder, y_train = encode_label(y_train, onehot = True)    # string label을 one hot vector로 변환
    encoder, y_val = encode_label(y_val, onehot = True)    # string label을 one hot vector로 변환

    os.environ["CUDA_VISIBLE_DEVICES"] = tc.GPU_INDEX   # gpu 번호 선택

    bondinet = Bondinet(n_classes = tc.N_CLASSES, weight_decay = tc.WEIGHT_DECAY)   # bondinet 모델 생성
    
    bondinet.train(X_train = X_train, y_train = y_train, 
                   X_val = X_val, y_val  = y_val,
                   n_epoch = tc.EPOCH, batch_size = tc.BATCH_SIZE, 
                   optimizer = tc.OPTIMIZER, momentum = tc.MOMENTUM, base_lr=tc.BASE_LR, 
                   lr_scheduler = tc.LR_SCHEDULER, lr_decay_interval = tc.LR_DECAY_INTERVAL, lr_decay_rate = tc.LR_DECAY_RATE,
                   model_dir_path = model_dir_path, batch_random_seed = tc.BATCH_RANDOM_SEED,
                   model_save_interval = tc.MODEL_SAVE_INTERVAL, log_write_interval = tc.LOG_WRITE_INTERVAL, resume = tc.RESUME_TRAIN)   # CNN 모델 학습

# 완료
# dresden, SMDB에 따라 이미지 데이터 분할되도록 수정 (ex. data/original/SMDB, data/dresden/original/Dresden)
# SMDB 사용하지 않는 모델, 확장자 제거
# 이벤트 파일 저장하도록 해서, tensorboard로 그래프 볼 수 있도록 수정
# dresden 이미지 데이터셋 프로젝트 path로 옮기기
# D70, D79s 인스턴스 번호 문제 해결하기
# dresden 데이터셋은 장면 분할 사용
# Train/Val/Test set 분할 
# 저장한 모델을 불러올 수 있도록 수정
# Dresden의 경우 scene_split 호출하도록 수정 (SINGLE_DEVICE 파라미터에 따라 다르게 split 되도록, SINGLE_DEVICE 파라미터 pickle 파일명에 포함되도록)
# tensorboard 히스토그램은 gradient, variable처럼 같이 나오도록 수정
# bondinet 생성자에 learning scheduler 사용 여부, optimizer의 종류 추가 하고, 모델 디렉토리명에 포함되도록 수정
# 사용되지 않는 파라미터가 전달된 경우, error 발생하도록 수정 (ex. LEARNING_SCHEDULER를 지정하지 않았는데, DECAY_RATE를 지정한 경우)
