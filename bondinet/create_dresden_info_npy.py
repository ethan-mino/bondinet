from tqdm import tqdm 
import csv
import os
from glob import glob
import re
import numpy as np
from itertools import groupby
import pprint
from config import CommonConfig as cc

DRESDEN_DIR_PATH = os.path.join(cc.PROJECT_PATH, cc.UNZIPED_ORIGINAL_IMG_DIR_PATH, "Dresden") 

MODEL_INDEX = 1
POSITION_INDEX = 3
MOTIVE_INDEX = 4

if __name__ == "__main__" : # dresden.csv 파일을 기반으로 dresden.npy 파일을 생성
    print(cc.DRESDEN_NPY_FILE_PATH, "will be removed. if you don't want to, close the program\n")
    
    dresden_info_list = set()    # csv 파일안의 정보 목록
    real_file_info_list = set()  # 실제 소유중인 이미지 파일의 정보

    f = lambda x, y, z : z
    real_file_list = [j for i in os.walk(DRESDEN_DIR_PATH) for j in f(*i)]    # 실제 소유중인 파일명 목록

    with open(cc.DRESDEN_CSV_FILE_PATH, 'r') as f_csv:   # Dresden.csv 파일 open
        reader = csv.reader(f_csv)

        csv_headers = next(reader)
        csv_rows = [csv_row for csv_row in reader]

        dresden_csv_file_list = []  # csv 파일안의 파일명 

        for file_info in tqdm(csv_rows):    # csv 파일의 각 row는 이미지 파일 정보
            brand, model, instance, position_num, motive_num, position_name, motive_name, filename, shot = file_info
            
            dresden_info_list.add(tuple(file_info)) # csv 파일의 모든 row 추가
            dresden_csv_file_list.append(filename)  # csv 파일안에 작성된 이미지 파일명 목록

            if filename in real_file_list : # 실제 소유중인 파일이라면
                real_file_info_list.add(tuple(file_info))   # 실제 소유중인 파일의 정보는 real_file_info에 추가


    real_file_set = set(real_file_list) # 실제 소유중인 이미지 파일 정보 set (집합 연산을 위해)
    dresden_csv_file_set = set(dresden_csv_file_list)   # csv 파일안에 작성된 이미지 파일 정보 set

    union = list(real_file_set | dresden_csv_file_set)  # 교집합
    amb = list(real_file_set - dresden_csv_file_set)    # csv 파일에는 작성되어있지 않지만, 실제 소유중인 이미지 파일명 목록
    bma = list(dresden_csv_file_set - real_file_set)    # csv 파일에는 작성되어있지만, 소유 x 파일명 목록

    if os.path.exists(cc.DRESDEN_NPY_FILE_PATH) : # npy 파일이 있다면 제거
        os.remove(cc.DRESDEN_NPY_FILE_PATH)
    
    assert(len(amb) == len(bma)== 0) # dresden.csv 파일에 작성된 이미지 중 실제 존재 하지 않는 이미지가 있는 경우 error

    print("# of image : ", len(real_file_list))


    np.save(cc.DRESDEN_NPY_FILE_PATH, np.array(real_file_list))


