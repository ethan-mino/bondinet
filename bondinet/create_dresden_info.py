from config import CommonConfig as cc


from tqdm import tqdm 
import csv
import os
import numpy as np

DRESDEN_DIR_PATH = os.path.join(cc.PROJECT_PATH, cc.UNZIPED_ORIGINAL_IMG_DIR_PATH, "DIDB") 

def scene_id(position_idx, motive_idx) :    
    return int(position_idx) * 10 + int(motive_idx)

if __name__ == "__main__" : # dresden.csv 파일을 기반으로 dresden.npy 파일을 생성
    print(cc.DRESDEN_NPY_FILE_PATH, "will be removed. if you don't want to, close the program\n")
    
    real_file_info_list = []  # 실제 소유중인 이미지 파일의 정보 목록
    dresden_file_info_list = [] # dresden.csv 파일에 작성된 이미지 파일의 정보 목록
    no_exists = [] # 존재하지 않는 파일 목록

    f = lambda x, y, z : list(map(lambda filename : os.path.join(x, filename), z))  # 파일명과 path 연결
    real_file_path_list = [j for i in os.walk(DRESDEN_DIR_PATH) for j in f(*i)]    # 실제 소유중인 파일의 path 목록
    real_file_name_list = list(map(lambda x : os.path.basename(x), real_file_path_list)) # # 실제 소유중인 파일명 목록

    with open(cc.DRESDEN_CSV_FILE_PATH, 'r') as f_csv:   # Dresden.csv 파일 open
        reader = csv.reader(f_csv)

        csv_headers = next(reader)
        csv_rows = [csv_row for csv_row in reader]

        dresden_csv_file_list = []  # csv 파일에 작성된 파일명 목록

        for file_info in tqdm(csv_rows):    # csv 파일의 각 row는 이미지 파일 정보
            brand, model, instance, position_num, motive_num, position_name, motive_name, filename, shot = file_info
            file_info = [brand, model, int(instance), int(position_num), int(motive_num), position_name, motive_name, filename, int(shot), scene_id(position_num, motive_num)]  # scene 식별자 정보 추가, int형으로 변환 (instance, position_num, motive_num, shot)
            
            dresden_csv_file_list.append(filename)  # csv 파일안에 작성된 이미지 파일명 목록
            dresden_file_info_list.append(file_info)

            try :
                file_index = real_file_name_list.index(filename)    # 실제 소유중인 파일이라면
                real_file_info_list.append(file_info + [real_file_path_list[file_index]]) # 실제 소유중인 파일의 정보는 real_file_info에 추가 (file_path)정보도 추가
            except ValueError : 
                no_exists.append(filename) # 존재하지 않는 파일이라면 no_exists에 추가

    real_file_set = set(real_file_name_list) # 실제 소유중인 이미지 파일명 목록 set (집합 연산을 위해)
    dresden_csv_file_set = set(dresden_csv_file_list)   # csv 파일안에 작성된 이미지 파일명 목록 set

    union = list(real_file_set | dresden_csv_file_set)  # csv 파일에는 작성되어있고, 실제 소유중인 파일명 목록
    amb = list(real_file_set - dresden_csv_file_set)    # csv 파일에는 작성되어있지 않지만, 실제 소유중인 이미지 파일명 목록
    bma = list(dresden_csv_file_set - real_file_set)    # csv 파일에는 작성되어있지만, 소유 x 파일명 목록

    if os.path.exists(cc.DRESDEN_NPY_FILE_PATH) : # npy 파일이 있다면 제거
        os.remove(cc.DRESDEN_NPY_FILE_PATH)
    
    assert(len(amb) == len(bma)== 0) # dresden.csv 파일에 작성된 이미지 중 실제 존재 하지 않는 이미지가 있는 경우 error

    print("# of image : ", len(real_file_info_list))
    
    np.save(cc.DRESDEN_NPY_FILE_PATH, np.array(real_file_info_list, dtype = "O"))    # 실제 소유중인 이미지 파일의 정보 목록을 npy 파일에 저장


