import os
import zipfile
import shutil
import joblib
import numpy as np
from tqdm import tqdm 
import traceback
from glob import glob
from PIL import Image
from sklearn.preprocessing import OneHotEncoder 
from sklearn.preprocessing import LabelEncoder
from patch_extractor import patch_extractor_one_arg
import re

# TODO : zip_dir 잘 동작하는지 확인
def zip_dir(target_dir_path, zip_file_path):
    with zipfile.ZipFile(zip_file_path,'w',zipfile.ZIP_DEFLATED) as zip : 
        for dirpath, dirnames, filenames in tqdm(os.walk(target_dir_path)):
            for filename in tqdm(filenames):
                zip.write(os.path.join(dirpath,filename))

def unzip(source_path, dest_path):  # https://gldmg.tistory.com/141
    error_members = []
    print(source_path, dest_path)
    if not os.path.exists(dest_path) :  # dest_path에 해당하는 파일/디렉터리가 존재하지 않는 경우에만 압축 해제
        with zipfile.ZipFile(source_path, 'r') as zf:
            zipInfo = zf.infolist()

            for member in tqdm(zipInfo):
                try : 
                    member.filename = member.filename.encode('cp437').decode('euc-kr', 'ignore')
                    #print(member.filename)
                    zf.extract(member, dest_path)
                finally : 
                    print(member.filename)
    else : 
        print("dest path is already exists!")

    print("unzip error: ", error_members)   
    return error_members    # 에러가 발생한 file path 목록 리턴

def center_crop(img, result_height, result_width) : # 이미지를 result_height * result_width 크기로 center_crop
    # TODO : 이미지의 높이 또는 너비가 홀수냐 아니냐에 따라 crop 결과 이미지 크기가 다른지 확인

    width, height = img.size

    left = (width - result_width) / 2
    top = (height - result_height) / 2
    right = (width + result_width) / 2
    bottom = (height + result_height) / 2

    return img.crop((left, top, right, bottom));

def encode_label(label, onehot = False) :
    if not onehot :
        encoder = LabelEncoder()
        transformed = encoder.fit_transform(label)
    else :
        encoder = OneHotEncoder()
        transformed = encoder.fit_transform(np.array(label).reshape(-1, 1)).toarray()
    return encoder, transformed

def decode_label(encoder, label) : 
    return encoder.inverse_transform(label)

def save_data(data, pickle_file_path) :
        joblib.dump(data, pickle_file_path)

def load_data(pickle_file_path) : # pickle file로부터 data load
    try :
        data = joblib.load(pickle_file_path)
        print("data loaded!")
        return data 
    except :    # pickle 파일에서 데이터를 load하는데 실패한 경우
        traceback.print_exc()
        print("\nfail to load data from pickle file\n")
        return None

def get_all_files(root_dir_path) :   # root_dir_path 하위의 모든 파일들의 path를 반환
    all_file_path = []

    for path, dirs, files in os.walk(root_dir_path) :
        for file in files : 
            all_file_path.append(os.path.join(path, file))

    return all_file_path;

def create_empty_dir(dir_path) :
    if os.path.exists(dir_path): # 디렉터리가 이미 존재하는 경우
        shutil.rmtree(dir_path) # 해당 디렉토리 삭제

    os.makedirs(dir_path)   # 디렉터리 생성

def img_to_array(img_dir_path_list, pickle_dir_path, option) : # 이미지가 저정된 디렉토리 하위에 저장된 이미지 데이터를 모두 불러와서 numpy array로 변환하는 함수 
    PICKLE_EXT = ".pkl"
    IMG_DATA_PICKLE_NAME = "img_data"    # 이미지 데이터와 레이블을 저장할 pickle 파일명
    PATCH_PICKLE_NAME = "patch"
    TEMP_FILE_NAME = "temp"

    def get_temp_file_list(dir_path, sort = True, reverse = False) : # dir_path 하위의 temp 파일 목록을 반환
        temp_file_pattern = f"temp_[0-9]+{PICKLE_EXT}"
        temp_file_list = [file_path for file_path in glob(os.path.join(dir_path, "*")) if re.search(temp_file_pattern, file_path)]    # dir_path 하위 파일 중 pattern에 일치하는 file만 걸러내어 temp 파일 목록을 구함.
    
        if sort and temp_file_list:
            temp_file_list.sort(reverse = True, key = lambda x : int(re.findall("[0-9]+", os.path.basename(x))[0]))  # 파일명에서 끝의 숫자만 남긴 후, int형으로 변환하여 정렬

        return temp_file_list  

    # option
    max_img_per_class = option["max_img_per_class"] # 모델 당 최대 이미지 개수
    resume = option["resume"]   # resume이 true이면, 저장된 모델 이후부터 변환
    
    if "center_crop" in option: # option dictionary에 "center_crop" key가 있는지 확인
        center_crop = True
        crop_width = option["center_crop"]["width"]
        crop_height = option["center_crop"]["height"]
    else : 
        center_crop = False
        crop_width, crop_height = [None, None]
    
    pickle_dir_path = os.path.join(pickle_dir_path, f"{crop_width}_{crop_height}_{max_img_per_class}") # 변환이 완료된 후 데이터를 저장할 디렉토리 path

    if "patch_option" in option :   # 이미지를 patch 단위로 나누는 경우
        patch_option = option["patch_option"]

        pickle_dir_path = f"{pickle_dir_path}_{patch_option['dim']}_{patch_option['offset']}_{patch_option['stride']}_{patch_option['rand']}_{patch_option['threshold']}_{patch_option['num']}" # 변환이 완료된 후 데이터를 저장할 디렉토리 path
        pickle_file_path = os.path.join(pickle_dir_path, f"{PATCH_PICKLE_NAME}{PICKLE_EXT}")  # 변환이 완료된 전체 데이터를 저장할 파일의 path
    else : 
        pickle_file_path = os.path.join(pickle_dir_path, f"{IMG_DATA_PICKLE_NAME}{PICKLE_EXT}")  # 변환이 완료된 전체 데이터를 저장할 파일의 path


    cur_model_index = 0  # 현재 처리중인 모델의 index
    error_file_name_list = []   # array로 변환하는데 실패한 이미지 파일명 목록
    X, y = [], []   # 이미지 데이터(array), label

    if resume == True : # resume 파라미터가 True인 경우, 데이터를 불러와 작업을 재개
        if os.path.exists(pickle_file_path) : # 전체 데이터를 저장한 파일이 있다면,
            data = load_data(pickle_file_path)  # pickle 파일에서 데이터를 불러옴

            if data is not None : 
                X = data["X"]
                y = data["y"]
                error_file_name_list = data["error_file_name_list"]
                return X, y, {"error_cnt" : len(error_file_name_list), "error_file_list" : error_file_name_list, "pickle_dir_path" : pickle_dir_path}   # array로 변환된 이미지 데이터, 레이블, 기타 정보 반환
        else : # 전체 데이터를 저장한 파일이 없다면,
            temp_file_list = get_temp_file_list(pickle_dir_path, sort = True, reverse = True) # temp 파일 리스트
        
            if temp_file_list : # temp 파일이 있다면
                last_temp_file = temp_file_list[0]  # 마지막 temp 파일 path
            
                data = load_data(last_temp_file)  # 마지막 temp 파일에서 데이터를 불러옴
                if data is not None :   # temp_file에서 데이터를 불러온 경우
                    cur_model_index = data["save_model_index"]  + 1 # 현재 처리중인 모델의 index 

    model_dir_path_list = [os.path.join(img_dir_path, img_path) for img_dir_path in img_dir_path_list for img_path in os.listdir(img_dir_path)]   # TODO : glob으로 변경, img_dir_path_list의 각각의 img_dir_path에 있는 파일 및 디렉토리의 path 목록(즉, 카메라 모델 디렉토리의 path 목록)
    n_model = len(model_dir_path_list)    # 카메라 모델 개수
    
    if not os.path.exists(pickle_dir_path): # 데이터를 저장할 디렉터리가 없다면 생성
        os.makedirs(pickle_dir_path)

    with tqdm(total = n_model, initial = cur_model_index, desc = "Load Img Data") as model_bar : 
        while cur_model_index < n_model  :
            processed_model_img_cnt = 0;    # 처리된 현재 카메라 모델의 이미지 개수
            cur_model_dir_path = model_dir_path_list[cur_model_index] # 현재 처리중인 모델 디렉토리의 경로
            all_files_path = get_all_files(cur_model_dir_path)  # 카메라 모델 디렉토리 하위의 모든 파일 path
            model_name = os.path.basename(cur_model_dir_path)   # 현재 처리중인 모델의 이름

            n_files = len(all_files_path)
            tqdm_total = n_files if max_img_per_class is None else min(max_img_per_class, n_files)  # max_img_per_class이면 tqdm의 total은 해당 모델의 총 파일 수, 그렇지 않으면 total은 max_img_per_class와 총 파일 개수 중 작은 것

            with tqdm(total = tqdm_total, desc = "Load Img File") as file_bar : 
                for file_path in all_files_path : # 현재 모델 디렉토리 하위의 모든 파일에 대해 반복
                    try :
                        with Image.open(file_path) as img : 
                            array_img = np.asarray(img) # img to numpy array

                            if center_crop :   # crop 파라미터가 True인 경우
                                img = center_crop(img, crop_height, crop_width)    # 이미지를 crop_height * crop_width 크기로 center_crop 
                    
                            if array_img is None :  # cv2.imread()의 결과가 None인 경우
                                raise Exception

                            if "patch_option" in option :
                                patch_option["img"] = array_img
                                patches = patch_extractor_one_arg(patch_option) # quality func 적용
                                n_patch = len(patches) # 추출된 patch의 개수
                        
                                X += patches
                                y += [model_name for i in range(n_patch)]   # 각 patch는 원본 이미지의 label을 상속
                            else : 
                                X.append(array_img) 
                                y.append(model_name)
                        
                            if max_img_per_class is not None and max_img_per_class < n_files :
                                file_bar.update()

                            processed_model_img_cnt += 1    # 처리된 현재 카메라 모델의 이미지 개수 증가

                    except Exception as err:
                        print(file_path)    # Invalid SOS parameters for sequential JPEG, Premature end of JPEG file 문제 해결
                        traceback.print_exc()  # 에러 내용 출력
                        error_file_name_list.append(os.path.basename(file_path))   # 이미지를 불러올 때 error가 발생한 경우 해당 파일명 저장

                    if max_img_per_class is None or max_img_per_class >= n_files : # max_img_per_class가 해당 모델의 파일 수보다 크거나 같은 경우
                        file_bar.update()

                    if max_img_per_class != None and processed_model_img_cnt >= max_img_per_class :   # 각 모델에 대해 max_img_per_class개의 이미지만 array로 변환
                        break;
        
            temp_file_path = os.path.join(pickle_dir_path, f"{TEMP_FILE_NAME}_{cur_model_index + 1}{PICKLE_EXT}") # 현재 모델의 데이터를 저장할 temp 파일 path
            save_data({"X" : X, "y" : y, "save_model_index" : cur_model_index, "error_file_name_list" : error_file_name_list}, temp_file_path)    # array로 변환된 이미지 데이터, 레이블, 저장 완료된 카메라 모델의 index를 pickle 파일에 저장
            X.clear(), y.clear(), error_file_name_list.clear()    # temp 파일에 저장한 후 데이터 비움
            
            cur_model_index += 1    # 현재 처리중인 모델의 index 증가
            model_bar.update()  # tqdm update
    
    temp_file_list = get_temp_file_list(pickle_dir_path, sort = True)   # temp 파일 path 목록
        
    for temp_file_path in temp_file_list : 
        data = load_data(temp_file_path)  # 각 temp 파일에서 데이터를 불러옴
        if data is not None : 
            X += data["X"]
            y += data["y"]
            error_file_name_list += data["error_file_name_list"]
        else : # temp 파일 중 하나라도 load에 실패한 경우 Error를 raise
            raise Exception(f"can't load data from {os.path.basename(temp_file)}")

    save_data({"X" : X, "y" : y, "error_file_name_list" : error_file_name_list}, pickle_file_path)    # 각 temp 파일을 합쳐서 pickle 파일에 저장.
        
    return X, y, {"error_cnt" : len(error_file_name_list), "error_file_list" : error_file_name_list, "pickle_dir_path" : pickle_dir_path} # array로 변환된 이미지 데이터, 레이블, 기타 정보 반환
        
