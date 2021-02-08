import data_processing
from config import CommonConfig 
from config import TrainConfig
from config import PreprocessingConfig

import os
import numpy as np

import pickle
import tensorflow as tf
from PIL import Image

# TODO : 함수 주석 달기 및 모듈화
# HEIC 이미지 파일 변환 -> HEIC 이미지는 사용하지 않음.

def unzip_all_img_file(skip = False) : 
    dest_dir_path = os.path.join(CommonConfig.PROJECT_PATH, CommonConfig.IMG_ROOT_DIR_NAME)   # 압축해제 된 이미지를 저장할 디렉토리의 경로

    data_processing.unzip_all([CommonConfig.ORIGINAL_PHONE_IMG_DIR_PATH, CommonConfig.ORIGINAL_CAMERA_IMG_DIR_PATH], dest_dir_path, skip = skip)


def load_img_data(img_dir_path_list, pickle_file_path, option) : # 이미지가 저정된 디렉토리 하위에 저장된 이미지 데이터를 모두 불러오는 함수 (pillow의 Image 사용)
    # option
    max_img_per_class = option["max_img_per_class"]
    resume = option["resume"]
    save = option["save"]
    img_width = option["crop"]["width"]
    img_height = option["crop"]["height"]

    next_img_index = 0 # array로 변환할 이미지의 index
    cur_model_index = 0  # 현재 처리중인 모델의 index
    error_file_name_list = []   # array로 변환하는데 실패한 이미지 파일명 목록
    X, y = [], []

    pickle_dir_path = os.path.dirname(pickle_file_path) # pickle 파일이 위치한 디렉토리의 path
    pickle_path_except_ext, pickle_ext = os.path.splitext(pickle_file_path) # pickle 파일의 확장자와 나머지 path를 분리
    pickle_file_path = os.path.join(pickle_dir_path, f"{pickle_path_except_ext}_{img_width}_{img_height}_{max_img_per_class}{pickle_ext}")  # pickle 파일명을 "pickle 파일명_너비_높이_모델당 이미지 개수.확장자"로 변경

    total_model_dir_path_list = [os.path.join(img_dir_path, img_path) for img_dir_path in img_dir_path_list for img_path in os.listdir(img_dir_path)]   # img_dir_path_list의 각각의 img_dir_path에 있는 파일 및 디렉토리의 path 목록(즉, 카메라 모델 디렉토리의 path 목록)

    if resume == True : # resume 파라미터가 True인 경우, pickle 파일에서 데이터를 불러와 작업을 재개
        data = data_processing.load_data(pickle_file_path)  # pickle 파일에서 데이터를 불러옴
        if data != None :   # pickle_file_path에 파일이 존재하는 경우
            X = data["X"]   # array로 변환된 이미지 데이터
            y = data["y"]   # label
            cur_model_index = data["save_model_index"]  + 1 # 현재 처리중인 모델의 index 
            next_img_index = data["next_img_index"] # array로 변환할 이미지의 index (단순 출력용)
            #print(X, y, save_model_index, next_img_index)

    while cur_model_index < len(total_model_dir_path_list) :    
        processed_model_img_cnt = 0;    # 처리된 현재 카메라 모델의 이미지 개수
        cur_model_dir_path = total_model_dir_path_list[cur_model_index] # 현재 처리중인 모델 디렉토리의 path
        all_files_path = data_processing.get_all_files(cur_model_dir_path)  # 카메라 모델 디렉토리 하위의 모든 파일 path
        model_name = os.path.basename(cur_model_dir_path)   # 현재 처리중인 모델의 이름

        print(model_name)   # 모델 이름 출력

        for file_path in all_files_path : # 현재 모델 디렉토리 하위의 모든 파일에 대해 반복
            try :
                img = Image.open(file_path) # Image 모듈을 이용하여 이미지 파일을 불러옴 (image를 많이 불러오면 too many open files 에러가 발생하므로 with 구문을 사용)


                croped_img = data_processing.center_crop(img, img_width, img_height)    # TODO : 이미지를 center_crop (일단 이미지 cropping, 후에 네트워크가 완성되면 (saturated pixel/ image dynamic 또는 quality function에 따른) patch priority 적용하는 걸로 변경)
                X.append(np.asarray(croped_img)) # 이미지를 numpy array로 변환
                y.append(model_name)
                processed_model_img_cnt += 1    # 처리된 현재 카메라 모델의 이미지 개수 증가
            except Exception as err: 
                print(err)  # 에러 내용 출력
                error_file_name_list.append(os.path.basename(file_path))   # 이미지를 불러올 때 error가 발생한 경우 해당 파일명 저장

            next_img_index += 1 # array로 변환할 이미지의 index
            print("next_img_index : " + str(next_img_index))

            if processed_model_img_cnt >= max_img_per_class :   # 각 모델에 대해 max_img_per_class개의 이미지만 array로 변환
                break;
        
        if save == True : # save 파라미터가 True(default)인 경우 pickle 파일에 저장
            data_processing.save_data({"X" : X, "y" : y, "save_model_index" : cur_model_index, "next_img_index" : next_img_index}, pickle_file_path)    # array로 변환된 이미지 데이터, 레이블, 저장 완료된 카메라 모델의 index를 pickle 파일에 저장
        cur_model_index += 1    # 현재 처리중인 모델의 index를 증가

    print("error_cnt : " + str(len(error_file_name_list)))  # image를 불러와 center crop하고, array로 변환할 때 에러가 발생한 파일의 개수 출력
    print(error_file_name_list) # 에러가 발생한 파일명 출력

    return X, y # array로 변환된 이미지 데이터와 레이블 반환



if __name__ == "__main__" : 
    img_data_pickle_path = os.path.join(CommonConfig.PROJECT_PATH, CommonConfig.IMG_DATA_PICKLE_NAME) # pickle 파일 path
    
    img_dir_path_list = [CommonConfig.PHONE_IMG_DIR_UNZIP_PATH, CommonConfig.CAMERA_IMG_DIR_UNZIP_PATH]

    unzip_all_img_file(skip = True) # 이미지 파일을 모두 unzip (skip 파라미터를 True로 지정하여 이미 파일이 존재하면 건너뜀)

    img_data, label = load_img_data(img_dir_path_list, img_data_pickle_path, 
                                    option = {"max_img_per_class" : PreprocessingConfig.MAX_IMG_PER_CLASS, 
                                              "resume" : PreprocessingConfig.RESUME, "save" : PreprocessingConfig.SAVE, 
                                              "crop" : {"width" : PreprocessingConfig.IMG_WIDTH, "height" : PreprocessingConfig.IMG_HEIGHT}})  # 이미지 데이터와 레이블 불러옴

    X = tf.placeholder(tf.float32, [None, 64, 64, 3])
    y = tf.placeholder(tf.float32, [None, TrainConfig.CLASS_NUM])

    tf.layers.conv2d(inputs = X, filters = 32, kernel_size = [4, 4, 3], padding = "VAILD", activation = None)

    tf.layers.conv2d(inputs = X, filters = 48, kernel_size = [5, 5, 32], padding = "VAILD", activation = None)

    tf.layers.conv2d(inputs = X, filters = 64, kernel_size = [5, 5, 48], padding = "VAILD", activation = None)

    tf.layers.conv2d(inputs = X, filters = 128, kernel_size = [5, 5, 64], padding = "VAILD", activation = None)



# 이미지 저장 시에는 .npy 또는 tensorflow의 record 사용
# 학습 시, keras의 generator 또는 tensorflow의 dictionary 사용

# TODO : 이미지를 numpy로 변환하지 않고, 이미지 그대로를 network에 입력하는 방법은?
# TODO : 이미지에서 64 * 64 크기의 패치 k를 추출
# TODO : saturated 픽셀이 있는 패치 제외
# TODO : 패치에 우선순위를 지정 (평균 값이 image dynamic의 절반에 가까운지에 따라)
# TODO : 각 패치에 전체 이미지와 동일한 레이블을 지정
# TODO : 각 입력 patch에서 훈련 세트에 대한 픽셀 단위 평균을 빼줌.
# TODO : dynamic을 줄이기 위해 진폭을 0.0125배로 조정


