from config import CommonConfig as cc
from config import PreprocessingConfig as pc

import sys
import numpy as np
from itertools import groupby

TRAIN_SCENE = 61
TEST_SCENE = 10
TEST_INSTANCE = 0

BRAND_INDEX = 0
MODEL_INDEX = 1
INSTANCE_NUM_INDEX = 2
POSITION_NUM_INDEX = 3
MOTIVE_NUM_INDEX = 4
POSITION_NAME_INDEX = 5
MOTIVE_NAME_INDEX = 6
FILE_NAME_INDEX = 7
SHOT_NUM_INDEX = 8
SCENE_ID_INDEX = 9
FILE_PATH_INDEX = 10

MERGE_SOURCE_MODEL = "D70s" # MERGE_TRAGET_MODEL으로 merge될 모델명
MERGE_TRAGET_MODEL = "D70"

def scene_split(single_device, seed) : 
    image_db = np.load(cc.DRESDEN_NPY_FILE_PATH)
    
    # 모델당 장면 개수 출력
    sort_by_model = lambda x : x[MODEL_INDEX] # 같은 키라도 분산되어있으면 한 그룹에 뭉쳐지지 않으므로 모델명을 기준으로 정렬하는 함수
    image_db = sorted(image_db, key = sort_by_model)  # 모델명을 기준으로 정렬하는 함수 (https://hulk89.github.io/python/2016/11/25/groupby/)
    
    model_info_dict = {} # 카메라 모델 정보

    for key, items in groupby(image_db, sort_by_model) :    # 모델별 정보로 변환
        # ex. { 'CoolPixS710': {'image_info_list' : [['Nikon', 'CoolPixS710', '1', '9', '1', 'Passageway Skulpturensammlung', 'Passageway view I', 'Nikon_CoolPixS710_1_13290.JPG', '13290', 19]]}, 
        #       'Agfa_DC-504_0' : {'image_info_list' : [['Nikon', 'CoolPixS710', '0', '12', '2', 'Home II', 'Computerfreak with his C64 and a plant', 'Nikon_CoolPixS710_0_12913.JPG', '12913', 12]]}}
        model_info_dict[key] = {"image_info_list" : [item for item in list(items)]} # image_info_list : 카메라 모델명이 'key'인 이미지들의 info 목록
    
    single_device_model_list = []
    
    for model_name, model_info in model_info_dict.items() : # 모델별 정보를 추가
        image_info_list = model_info["image_info_list"] 
        
        unique_scene = np.unique([image_info[SCENE_ID_INDEX] for image_info in image_info_list])   
        unique_instance = np.unique([image_info[INSTANCE_NUM_INDEX] for image_info in image_info_list])   
        
        n_scene = len(unique_scene) 
        n_instance = len(unique_instance)

        if n_instance <= 1 : 
            single_device_model_list.append(model_name)

        model_info_dict[model_name]["n_scene"] = n_scene    # 해당 모델의 scene 개수 정보 추가
        model_info_dict[model_name]["n_instance"] = n_instance    # 해당 모델의 인스턴스 개수 정보 추가

        model_info_dict[model_name]["unique_scene"] = unique_scene  # 해당 모델의 고유한 scene id 목록 정보 추가
    
    # Merge Nikon_D70(s)
    merge_target_model_n_instance = model_info_dict[MERGE_TRAGET_MODEL]["n_instance"] # merge target 모델의 인스턴스 개수
    
    for image_info in model_info_dict[MERGE_SOURCE_MODEL]["image_info_list"] : 
        image_info[MODEL_INDEX] = MERGE_TRAGET_MODEL  # 모델명이 D70s인 이미지 정보의 모델명을 D70으로 변경
        #image_info[INSTANCE_NUM_INDEX] = image_info[INSTANCE_NUM_INDEX] + merge_target_model_n_instance   # ex. merge target 모델의 인스턴스가 2개라면, merge source 모델의 인스턴스 번호에 2를 더해줌. TODO 

    model_info_dict[MERGE_TRAGET_MODEL]["image_info_list"] += model_info_dict[MERGE_SOURCE_MODEL]["image_info_list"] # D70s와 D70의 이미지 정보를 합침
    del model_info_dict[MERGE_SOURCE_MODEL]

    if not pc.SINGLE_DEVICE:    # pc.SINGLE_DEVICE에 따라 장치가 하나인 모델 정보 제거
        for single_device_model in single_device_model_list : 
            del model_info_dict[single_device_model]

    unique_scene_id_list = sorted(np.unique([j for i in model_info_dict.items() for j in (lambda key, value : value)(*i)["unique_scene"]]))  # unique한 scene 식별자 목록
    image_info_list = [j for i in model_info_dict.items() for j in (lambda key, value : value)(*i)["image_info_list"]]   # train/val/test set로 사용될 이미지들의 정보
    classes = model_info_dict.keys()    # 카메라 모델명 목록
    n_classes = len(classes)    # 클래스의 개수
    
    i = 0
    for class_num, class_name in zip(range(n_classes), classes) : 
        
        for image_info in image_info_list : 
            if image_info[MODEL_INDEX] == class_name : 
                np.append(image_info, class_num) # 각 이미지 정보에 클래스 번호 정보를 추가
                i += 1
            
    assert(len(image_info_list) == i)    # 클래스 번호가 지정되지 않은 파일이 있다면 error를 발생
    

    # Train, Val, Test 분할
    np.random.seed(seed)
    
    train_set, val_set, test_set = [], [], []
    
    print("# of Total Image : ", len(image_info_list)) # 남은 전체 이미지 개수 출력
    
    """
    # 장면당 이미지 개수 출력
    sorted_by_scene = sorted(image_db, key = lambda x : x[SCENE_ID_INDEX])  # 장면을 기준으로 정렬 (https://hulk89.github.io/python/2016/11/25/groupby/)
    
    for key, items in groupby(sorted_by_scene, lambda x : x[SCENE_ID_INDEX]) :    # 장면별 정보로 변환
        print("n_scene : ", key, "# : ", len([item for item in list(items)]))    # 장면당 이미지 개수 출력
    """

    random_scene_ids = np.random.permutation(unique_scene_id_list)  # scene_id 섞음
    test_scene_ids = random_scene_ids[-TEST_SCENE : ]   # test set에 할당된 scene 개수만큼 scene 식별자 추출
    
    for test_scene_id in test_scene_ids : 
        test_set += [image_info for image_info in image_info_list if image_info[SCENE_ID_INDEX] == test_scene_id and image_info[INSTANCE_NUM_INDEX] == TEST_INSTANCE]   # 인스턴스 번호가 TEST_INSTANCE이고, scene 식별자가 test_scene_ids에 속한 이미지만 

    train_scene_ids = random_scene_ids[:TRAIN_SCENE]  # train set에 할당된 scene 개수만큼 scene 식별자 추출
    for train_scene_id in train_scene_ids : 
        train_set += [image_info for image_info in image_info_list if image_info[SCENE_ID_INDEX] == train_scene_id and image_info[INSTANCE_NUM_INDEX] != TEST_INSTANCE]   # 인스턴스 번호가 TEST_INSTANCE가 아닌 이미지 중에서, scene 식별자가 train_scene_ids에 속한 이미지만 

    val_scene_ids = random_scene_ids[TRAIN_SCENE : -TEST_SCENE]  # train set에 할당된 scene 개수만큼 scene 식별자 추출
    for val_scene_id in val_scene_ids : 
        val_set += [image_info for image_info in image_info_list if image_info[SCENE_ID_INDEX] == val_scene_id and image_info[INSTANCE_NUM_INDEX] != TEST_INSTANCE]   # 인스턴스 번호가 TEST_INSTANCE가 아닌 이미지 중에서, scene 식별자가 val_scene_ids에 속한 이미지만 

    for class_name in classes : 
        assert class_name in [image_info[MODEL_INDEX] for image_info in train_set], f"{class_name} is not included in the train set."   # train_set에 누락된 카메라 모델이 있는지 확인
        assert class_name in [image_info[MODEL_INDEX] for image_info in test_set], f"{class_name} is not included in the test set."
        assert class_name in [image_info[MODEL_INDEX] for image_info in val_set], f"{class_name} is not included in the val set."

    unassigned = list(set(map(tuple, image_info_list)) - set(map(tuple, train_set)) - set(map(tuple, val_set)) - set(map(tuple, test_set)))   # train, validation, test set에 포함되지 않은 이미지
    
    print("Unassigned : ", len(unassigned))
    print("Train shot : ", len(train_set))
    print("Val shot : ", len(val_set))
    print("Test shot : ", len(test_set))
    
    assert len(image_info_list), len(unassigned) + len(train_set) + len(val_set) + len(test_set)    # unassigned, test, val, test 어디에도 포함되지 않은 이미지가 있는지 확인

    X_train, y_train = np.array(train_set)[:, FILE_PATH_INDEX], np.array(train_set)[:, MODEL_INDEX]
    X_val, y_val = np.array(val_set)[:, FILE_PATH_INDEX], np.array(val_set)[:, MODEL_INDEX]
    X_test, y_test = np.array(test_set)[:, FILE_PATH_INDEX], np.array(test_set)[:, MODEL_INDEX]    
    X_unassigned, y_unassigned = np.array(unassigned)[:, FILE_PATH_INDEX], np.array(unassigned)[:, MODEL_INDEX]

    return {"X_train" : X_train, "y_train" : y_train, 
            "X_val" : X_val, "y_val" : y_val, 
            "X_test" : X_test, "y_test" : y_test, 
            "X_unassigned" : X_unassigned, "y_unassigned" : y_unassigned}
