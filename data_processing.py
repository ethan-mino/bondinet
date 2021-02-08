import os
import zipfile
import shutil
import pickle

def unzip_all(source_dir_path_list, dest_dir_path, skip = False) :    # source_dir_path_list 안에 있는 각각의 디렉토리 내 zip 파일을 압축 해제하여 dest_dir_path에 저장
                                                        # source_dir_path_list : zip 파일을 저장하고 있는 디렉토리
                                                        # dest_dir_path : 압축 해제 결과를 저장할 디렉토리의 경로

    for source_rdir_path in source_dir_path_list :   # zip 파일을 저장하고 있는 각각의 source 디렉토리에 대해 반복
        source_dir_name = os.path.basename(source_rdir_path) # zip 파일을 저장하고 있는 디렉토리명
        file_name_list = os.listdir(source_rdir_path) # 해당 디렉토리 안의 파일명 리스트
        
        zip_file_name_list = [file_name for file_name in file_name_list if zipfile.is_zipfile(os.path.join(source_rdir_path, file_name))]   # 파일 중 zip 파일만 filter

        for zip_file_name in zip_file_name_list :    # 각 zip 파일을 압축 해제 
            print(zip_file_name)    # zip 파일명 출력

            zip_file_path = os.path.join(source_rdir_path, zip_file_name)    # 압축 해제 할 zip 파일의 경로
            zip_file_basename, ext = os.path.splitext(zip_file_name) # zip 파일명에서 basename과 확장자를 분리
            
            extract_dest_dir_path = os.path.join(dest_dir_path, source_dir_name, zip_file_basename)    # 압축 해제 한 파일들을 저장할 경로

            if os.path.exists(extract_dest_dir_path) and skip == True : # 이미 압축 해제된 디렉토리가 있고, skip 파라미터가 true인 경우 해당 zip파일을 압축 해제하지 않음.
                continue

            with zipfile.ZipFile(zip_file_path) as zip :    # zip 파일 압축 해제
                zip.extractall(path = extract_dest_dir_path)
    print("Data unzip complete!\n")

def center_crop(img, result_width, result_height) :
    # TODO : 이미지의 높이 또는 너비가 홀수냐 아니냐에 따라 crop 결과 이미지 크기가 다른지 확인

    width, height = img.size

    left = (width - result_width) / 2
    top = (height - result_height) / 2
    right = (width + result_width) / 2
    bottom = (height + result_height) / 2

    return img.crop((left, top, right, bottom));

def save_data(data, pickle_file_path) :
    with open(pickle_file_path, "wb") as img_data_pickle : 
        pickle.dump(data, img_data_pickle, protocol=pickle.HIGHEST_PROTOCOL)    # The advantage of HIGHEST_PROTOCOL is that files get smaller. This makes unpickling sometimes much faster.

def load_data(pickle_file_path) : 
    try :
        with open(pickle_file_path, "rb") as img_data_pickle : 
            data = pickle.load(img_data_pickle)
        print("data loaded!")
        return data 
    except :    # pickle 파일에서 데이터를 load하는데 실패한 경우
        print("\nfail to load data from pickle file\n")
        return None

def get_all_files(root_dir_path) :   # root_dir_path 하위의 모든 파일들의 path를 반환
    all_file_path = []

    for path, dirs, files in os.walk(root_dir_path ) :
        for file in files : 
            all_file_path.append(os.path.join(path, file))

    return all_file_path;

