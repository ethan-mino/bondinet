import os
import zipfile
import shutil
import joblib
from sklearn.preprocessing import OneHotEncoder 
from sklearn.preprocessing import LabelEncoder
import numpy as np
from tqdm import tqdm 
import traceback

def unzip(source_file, dest_path):  # https://gldmg.tistory.com/141
    with zipfile.ZipFile(source_file, 'r') as zf:
        zipInfo = zf.infolist()
        error_members = []

        for member in zipInfo:
            try:
                #print(member.filename.encode('cp437').decode('euc-kr', 'ignore'))
                member.filename = member.filename.encode('cp437').decode('euc-kr', 'ignore')
                zf.extract(member, dest_path)
            except Exception as err:
                print(err)
                error_members.append(source_file)

    return error_members    # 에러가 발생한 file path 목록 리턴

def unzip_all(source_dir_path_list, dest_dir_path, skip = False) :    # source_dir_path_list 안에 있는 각각의 디렉토리 내 zip 파일을 압축 해제하여 dest_dir_path에 저장
                                                        # source_dir_path_list : zip 파일을 저장하고 있는 디렉토리
                                                        # dest_dir_path : 압축 해제 결과를 저장할 디렉토리의 경로
    total_error_members = []
    
    for source_rdir_path in tqdm(source_dir_path_list, desc = "Unzip dir"):   # zip 파일을 저장하고 있는 각각의 source 디렉토리에 대해 반복
        source_dir_name = os.path.basename(source_rdir_path) # zip 파일을 저장하고 있는 디렉토리명
        file_name_list = os.listdir(source_rdir_path) # 해당 디렉토리 안의 파일명 리스트
        
        zip_file_name_list = [file_name for file_name in file_name_list if zipfile.is_zipfile(os.path.join(source_rdir_path, file_name))]   # 파일 중 zip 파일만 filter
        
        for zip_file_name in tqdm(zip_file_name_list, desc = "Unzip file") :    # 각 zip 파일을 압축 해제 
            zip_file_path = os.path.join(source_rdir_path, zip_file_name)    # 압축 해제 할 zip 파일의 경로
            zip_file_basename, ext = os.path.splitext(zip_file_name) # zip 파일명에서 basename과 확장자를 분리
            
            extract_dest_dir_path = os.path.join(dest_dir_path, source_dir_name, zip_file_basename)    # 압축 해제 한 파일들을 저장할 경로

            if os.path.exists(extract_dest_dir_path) and skip == True : # 이미 압축 해제된 디렉토리가 있고, skip 파라미터가 true인 경우 해당 zip파일을 압축 해제하지 않음.
                continue

            total_error_members += unzip(zip_file_path, extract_dest_dir_path) # zip 파일 압축 해제
    
    print("unzip error: ", total_error_members)
    print("Data unzip complete!\n")

def center_crop(img, result_height, result_width) : # 이미지를 result_height * result_width 크기로 center_crop
    # TODO : 이미지의 높이 또는 너비가 홀수냐 아니냐에 따라 crop 결과 이미지 크기가 다른지 확인

    width, height = img.size

    left = (width - result_width) / 2
    top = (height - result_height) / 2
    right = (width + result_width) / 2
    bottom = (height + result_height) / 2

    return img.crop((left, top, right, bottom));

def label_to_number(label, onehot = False) :
    if not onehot :
        return LabelEncoder().fit_transform(label)
    else :
        return OneHotEncoder().fit_transform(np.array(label).reshape(-1, 1)).toarray()

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
        
