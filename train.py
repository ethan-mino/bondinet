import data_processing
from patch_extractor import patch_extractor_one_arg
from config import CommonConfig as cc
from config import TrainConfig as tc
from config import PreprocessingConfig as pc

from functools import partial
import os
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from PIL import Image

# TODO : 함수 주석 달기 및 모듈화
# TODO : HEIC 이미지 파일 변환 -> HEIC 이미지는 사용하지 않음.
# TODO : 이미지를 numpy로 변환하지 않고, 이미지 그대로를 network에 입력하는 방법은?
# TODO : 이미지에서 64 * 64 크기의 패치 k를 추출
# TODO : saturated 픽셀이 있는 패치 제외
# TODO : 패치에 우선순위를 지정 (평균 값이 image dynamic의 절반에 가까운지에 따라)
# TODO : 각 패치에 전체 이미지와 동일한 레이블을 지정
# TODO : 각 입력 patch에서 훈련 세트에 대한 픽셀 단위 평균을 빼줌.
# TODO : dynamic을 줄이기 위해 진폭을 0.0125배로 조정
# TODO : 초기화 함수 지정
# TODO : Galaxy Note9 SM-N960N는 제외한 후 실험
# TODO : Train/Test set 분할 (모델마다 섞은 후 20퍼센트)

# 창희님 네트워크
# 원본 이미지를 겹치지 않게 256 * 256 패치 추출, 모두 사용
# 각 패치에 대해 분류, 패치들의 평균 정확도를 사용

# 실험
# 이미지 저장 시에는 .npy 또는 tensorflow의 record 사용
# 학습 시, keras의 generator 또는 tensorflow의 dictionary 사용
# 첫 에폭 로스가 대략 log2(클래스 개수)정도면 나쁜 네트워크는 아님
# 스마트폰 카메라 이미지만 사용?

# TODO : 다음 세미나 때까지 실험
# 1. non overlapping/overlapping, 패치 개수 달리해서 실험 (overlapping의 경우 패치 개수 많이 사용, non overlapping/overlapping는 네트워크 하나에 대해 실험)
# 2. 두 개 네트워크 전처리 맞추기 X
# 3. 0 ~ 1 사이의 값이 나오는지, flat/saturated pixel/ standard deviation 낮은 이미지 score 낮은지 확인
# 4. 이미지 전체의 퀄리티 점수가 낮은 이미지의 분류 성능은?
# 5. 본디넷 구현 세미나할 때, ovo/ova 좀 더 자세하게 설명
# 6. 장치 개수 상관없이 모든 모델을 사용한 거 / 모델 당 장치 개수 2개 이상인 모델만 사용한 거 실험

def unzip_all_img_file(skip = False) : 
    dest_dir_path = os.path.join(cc.PROJECT_PATH, cc.IMG_ROOT_DIR_NAME)   # 압축해제 된 이미지를 저장할 디렉토리의 경로

    data_processing.unzip_all([cc.ORIGINAL_PHONE_IMG_DIR_PATH, cc.ORIGINAL_CAMERA_IMG_DIR_PATH], dest_dir_path, skip = skip)


def load_img_data(img_dir_path_list, option) : # 이미지가 저정된 디렉토리 하위에 저장된 이미지 데이터를 모두 불러오는 함수 (pillow의 Image 사용)
    # option
    max_img_per_class = option["max_img_per_class"]
    resume = option["resume"]
    save = option["save"]
    
    if "center_crop" in option: # option dictionary에 "center_crop" key가 있는지 확인
        center_crop = True
        crop_width = option["center_crop"]["width"]
        crop_height = option["center_crop"]["height"]
    else : 
        center_crop = False
        crop_width, crop_height = [None, None]
    
    cur_model_index = 0  # 현재 처리중인 모델의 index
    error_file_name_list = []   # array로 변환하는데 실패한 이미지 파일명 목록
    X, y = [], []

    pickle_dir_path = os.path.join(cc.PROJECT_PATH, f"{crop_width}_{crop_height}_{max_img_per_class}") # pickle 파일이 위치한 디렉토리의 path ("center crop 너비_높이_모델당 이미지 개수")
    pickle_file_path = os.path.join(pickle_dir_path, f"{cc.IMG_DATA_PICKLE_NAME}{cc.PICKLE_EXT}")  # pickle 파일 path 

    total_model_dir_path_list = [os.path.join(img_dir_path, img_path) for img_dir_path in img_dir_path_list for img_path in os.listdir(img_dir_path)]   # img_dir_path_list의 각각의 img_dir_path에 있는 파일 및 디렉토리의 path 목록(즉, 카메라 모델 디렉토리의 path 목록)

    if resume == True : # resume 파라미터가 True인 경우, pickle 파일에서 데이터를 불러와 작업을 재개
        data = data_processing.load_data(pickle_file_path)  # pickle 파일에서 데이터를 불러옴
        if data != None :   # pickle_file_path에 파일이 존재하는 경우
            X = data["X"]   # array로 변환된 이미지 데이터
            y = data["y"]   # label
            cur_model_index = data["save_model_index"]  + 1 # 현재 처리중인 모델의 index 
            error_file_name_list = data["error_file_name_list"] # 현재 처리중인 모델의 index 
    
    n_model = len(total_model_dir_path_list)    # 카메라 모델 개수
    
    if not os.path.exists(pickle_dir_path): # 데이터를 저장할 디렉터리가 없다면 생성
        os.makedirs(pickle_dir_path)

    with tqdm(total = n_model, initial = cur_model_index, desc = "Load Img Data") as model_bar : 
        while cur_model_index < n_model  :
            processed_model_img_cnt = 0;    # 처리된 현재 카메라 모델의 이미지 개수
            cur_model_dir_path = total_model_dir_path_list[cur_model_index] # 현재 처리중인 모델 디렉토리의 경로
            all_files_path = data_processing.get_all_files(cur_model_dir_path)  # 카메라 모델 디렉토리 하위의 모든 파일 path
            model_name = os.path.basename(cur_model_dir_path)   # 현재 처리중인 모델의 이름
        
            for file_path in all_files_path : # 현재 모델 디렉토리 하위의 모든 파일에 대해 반복
                try :
                    img = Image.open(file_path) # Image 모듈을 이용하여 이미지 파일을 불러옴 (image를 많이 불러오면 too many open files 에러가 발생하므로 with 구문을 사용)

                    if center_crop :   # crop 파라미터가 True인 경우
                        img = data_processing.center_crop(img, crop_height, crop_width)    # TODO : 이미지를 center_crop (일단 이미지 cropping, 후에 네트워크가 완성되면 (saturated pixel/ image dynamic 또는 quality function에 따른) patch priority 적용하는 걸로 변경)

                    X.append(np.asarray(img)) # 이미지를 numpy array로 변환
                    y.append(model_name)
                    processed_model_img_cnt += 1    # 처리된 현재 카메라 모델의 이미지 개수 증가
                except Exception as err: 
                    print(err)  # 에러 내용 출력
                    error_file_name_list.append(os.path.basename(file_path))   # 이미지를 불러올 때 error가 발생한 경우 해당 파일명 저장

                if max_img_per_class != None and processed_model_img_cnt >= max_img_per_class :   # 각 모델에 대해 max_img_per_class개의 이미지만 array로 변환
                    break;
        
            if save == True and cur_model_index % 10 == 0: # save 파라미터가 True(default)인 경우 pickle 파일에 저장
                data_processing.save_data({"X" : X, "y" : y, "save_model_index" : cur_model_index, "error_file_name_list" : error_file_name_list}, pickle_file_path)    # array로 변환된 이미지 데이터, 레이블, 저장 완료된 카메라 모델의 index를 pickle 파일에 저장
            
            cur_model_index += 1    # 현재 처리중인 모델의 index 증가
            model_bar.update()  # tqdm update
    
    print("load error_cnt : " + str(len(error_file_name_list)))  # image를 불러와 center crop하고, array로 변환할 때 에러가 발생한 파일의 개수 출력
    print(error_file_name_list) # 에러가 발생한 파일명 출력

    if "patch_option" in option :   # patch_option 파라미터가 있는 경우 각 이미지에서 patch를 추출
        patch_option = option["patch_option"]
        patch_save_interval = patch_option.pop("save_interval") # patch를 파일에 저장하는 간격(이미지 기준)
        patch_file_path = os.path.join(pickle_dir_path, f"{cc.PATCH_PICKLE_NAME}_{patch_option.dim}_{patch_option.offset}_{patch_option.stride}_{patch_option.rand}_{patch_option.threshold}_{patch_option.num}{cc.PATCH_EXT}") # patch를 저장하는 pickle 파일의 경로
        
        cur_img_index = 0 # 현재 patch를 추출중인 이미지의 index
        extract_error_cnt = 0   # patch 추출 실패 횟수

        total_patch, patch_label = [], []
        n_img = len(X) # patch를 추출할 이미지 개수 

        if resume : 
            patch_data = data_processing.load_data(patch_file_path)  # pickle 파일에서 patch 데이터를 불러옴
            if patch_data != None :   # patch_file_path에 파일이 존재하는 경우
                cur_img_index = data["save_img_index"] + 1
                extract_error_cnt = data["extract_error_cnt"]
                total_patch =  data["total_patch"]
                patch_label = data["patch_label"]

        with tqdm(total = n_img, initial = cur_img_index, desc = "Extract patch") as patch_bar : 
            while cur_img_index < n_img :
                img = X[cur_img_index]  # patch를 추출할 이미지
                img_label = y[cur_img_index]    # 이미지의 label

                patch_option["img"] = img
                patches = patch_extractor_one_arg(patch_option) # 각 이미지의 patch를 추출
                total_patch.append(patches)  
                
                n_patch = len(patches) # 추출된 patch의 개수
                patch_label.full((1, n_patch), img_label)   # 원본 이미지의 label을 상속

                if save == True and cur_img_index % patch_save_interval == 0:   # save 파라미터가 True인 경우 이미지 patch_save_interval 간격으로 patch를 저장
                    data_processing.save_data({"total_patch" : total_patch, "patch_label" : patch_label, "extract_error_cnt" : extract_error_cnt, "save_img_index" : cur_img_index}, patch_file_path)    
                
                cur_img_index += 1
                patch_bar.update()

    else : 
        return X, y # array로 변환된 이미지 데이터와 레이블 반환



class CNN : # https://minimin2.tistory.com/36
    learning_schedule_name = "learning_schedule"

    def __init__(self) : 
        self.session = tf.Session() # TODO : 저장할 모델을 불러올 수 있도록 수정

    def build_net(self, decay_steps) :
        self.X = tf.placeholder(tf.float32, [None, 64, 64, 3])
        self.y = tf.placeholder(tf.int32, [None, tc.CLASS_NUM])
        
        l2_regularizer = tf.contrib.layers.l2_regularizer(tc.WEIGHT_DECAY)
        custom_conv2d = partial(tf.layers.conv2d, activation = tc.CONV_ACTIVATION, kernel_initializer = tc.CONV_KERNEL_INITIALIZER, bias_initializer = tc.CONV_BIAS_INITIALIZER, use_bias = tc.USE_CONV_BIAS, padding = tc.CONV_PADDING, kernel_regularizer = l2_regularizer)
        custom_max_pooling2d = partial(tf.layers.max_pooling2d, pool_size = tc.MAX_POOLING_POOL_SIZE, strides = tc.MAX_POOLING_STRIDES, padding = tc.MAX_POLLING_PADDING)
        custom_dense = partial(tf.layers.dense, kernel_regularizer = l2_regularizer)

        l1_conv = custom_conv2d(inputs = self.X, filters = tc.L1_CONV_FILTERS, kernel_size = tc.L1_CONV_KERNEL_SIZE)
        l1_max_pool = custom_max_pooling2d(l1_conv)
    
        l2_conv = custom_conv2d(inputs = l1_max_pool, filters = tc.L2_CONV_FILTERS, kernel_size = tc.L2_CONV_KERNEL_SIZE)
        l2_max_pool = custom_max_pooling2d(l2_conv)
    
        l3_conv = custom_conv2d(inputs = l2_max_pool, filters = tc.L3_CONV_FILTERS, kernel_size = tc.L3_CONV_KERNEL_SIZE)
        l3_max_pool = custom_max_pooling2d(l3_conv)

        l4_conv = custom_conv2d(inputs = l3_max_pool, filters = tc.L4_CONV_FILTERS, kernel_size = tc.L4_CONV_KERNEL_SIZE)

        l4_flatten = tf.layers.flatten(l4_conv) # TODO : 논문 코드에는 flatten 과정이 없는데, IP1 layer에 flatten하여 입력하는 것이 맞는지 확인
        self.ip1 = custom_dense(inputs = l4_flatten, units = tc.IP1_UNITS, activation = tc.IP1_ACTIVATION_FUNC)
        
        self.logits = custom_dense(inputs = self.ip1, units = tc.IP2_UNITS, activation = tc.IP2_ACTIVATION_FUNC)
        self.base_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = self.y, logits = self.logits)) # https://stats.stackexchange.com/questions/327348/how-is-softmax-cross-entropy-with-logits-different-from-softmax-cross-entropy-wi
        
        regularization_losses =  tf.losses.get_regularization_losses()
        self.reg_loss = tf.add_n([self.base_loss] + regularization_losses)    # l2 regulariztion loss (https://stackoverflow.com/questions/37107223/how-to-add-regularizations-in-tensorflow, http://blog.naver.com/PostView.nhn?blogId=infoefficien&logNo=221143520556)

        global_step = global_step = tf.Variable(0, trainable=False)

        # learning_schedule = tf.train.exponential_decay(learning_rate = tc.BASE_LR, global_step = global_step, decay_steps = decay_steps, decay_rate = tc.LR_DECAY_RATE, staircase = True, name = self.learning_schedule_name)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.001)  # TODO : leaning_schedule 지정, weight_decay 적용할 수 있도록 optimizer 변경, sgd로 변경
        self.training_op = self.optimizer.minimize(self.reg_loss)
        # self.predicted = tf.argmax(self.logits, 1)    # what is parameter 1 - axis

    def train(self, X_data, y_data, n_epoch, batch_size) :
        np.random.seed(tc.BATCH_RANDOM_SEED)    # randint SEED 지정

        n_examples = len(X_data)  # 이미지의 개수
        n_iteration = int(n_examples / batch_size) # iteration = 이미지 개수 / batch size
        
        decay_steps = tc.LR_DECAY_INTERVAL * n_iteration
        self.build_net(decay_steps)
        
        self.session.run(tf.global_variables_initializer()) # 변수 초기화
        
        """
        if use_learning_rate_decay : 
            learning_rate = tf.get_default_graph().get_tensor_by_name(f"{self.learning_schedule_name}:0")
            print(tf.train.exponential_decay(learning_rate = tc.BASE_LR, global_step = self.global_step, decay_steps = tc.LR_DECAY_INTERVAL * n_iteration, decay_rate = tc.LR_DECAY_RATE, staircase = True))
            tf.assign(learning_rate, tf.train.exponential_decay(learning_rate = tc.BASE_LR, global_step = self.global_step, decay_steps = tc.LR_DECAY_INTERVAL * n_iteration, decay_rate = tc.LR_DECAY_RATE, staircase = True))  # TODO : learning_rate를 base로 지정해줘도 괜찮을지 생각.        
        """

        model_file_dir_path = os.path.join(cc.PROJECT_PATH, f"{tc.MODEL_FILE_NAME}_{pc.IMG_CROP_WIDTH}_{pc.IMG_CROP_HEIGHT}_{pc.MAX_IMG_PER_CLASS}")  # 모델을 저장할 디렉토리의 경로
        model_file_path = os.path.join(model_file_dir_path, tc.MODEL_FILE_NAME) # 모델을 저장할 파일의 경로
        log_dir_path = os.path.join(model_file_dir_path, tc.LOG_DIR_NAME)

        data_processing.create_empty_dir(model_file_dir_path) # 모델 파일을 저장할 디렉토리가 없다면 생성, 이미 있다면 디렉토리를 비움

        loss_sum = 0    # 배치당 정확도를 모두 더한 값을 저장

        saver = tf.train.Saver()    # 학습된 모델을 저장할 Savar 객체
        # file_writer = tf.summary.FileWriter(log_dir_path, tf.compat.v1.get_default_graph())
        
        for cur_epoch in tqdm(range(n_epoch), desc = "Epoch") :
            loss_sum = 0 # epoch의 배치당 loss 총합

            for cur_iteration in range(n_iteration) :
                shuffled_indices = np.random.randint(0, n_examples, batch_size) # 0 ~ (이미지 데이터 개수 -1) 사이의 수를 batch size만큼 random 추출
                
                # 모든 batch의 loss / 배치 개수로 loss 계산
                X_batch = np.array(X_data)[shuffled_indices]  # random image data batch
                y_batch = np.array(y_data)[shuffled_indices]  # random label batch
                
                cur_loss, _ = self.session.run([self.reg_loss, self.training_op], feed_dict = {self.X : X_batch / 255, self.y : y_batch})   
                
                """ TODO : tensorboard 코드 수정 
                if cur_iteration % tc.LOG_WRITE_INTERVAL == 0 : 
                    steps = cur_epoch * n_iteration * cur_iteration
                    file_writer.add_summary(cur_loss, steps)
                """

                loss_sum += cur_loss
            print(f"{cur_epoch + 1}th Epoch Avg loss : {loss_sum  / n_iteration}")


            if cur_epoch % tc.MODEL_SAVE_INTERVAL == 0 :    # MODEL_SAVE_INTERVAL epoch마다 모델을 save
                saver.save(self.session, model_file_path) 
            
        saver.save(self.session, model_file_path) # 모델 저장

    def extract_feature_vectors(self, X_data) : # relu layer에서 forward propagation을 stop하여 feature 벡터 추출
        return self.session.run(self.ip1, feed_dict = {self.X : X_data})

    def predict(self, X_data) :
        return self.session.run(self.predicted, feed_dict = {self.X : X_data})

if __name__ == "__main__" : 
    img_dir_path_list = [cc.PHONE_IMG_DIR_UNZIP_PATH] # 스마트폰 이미지만 사용

    unzip_all_img_file(skip = True) # 이미지 파일을 모두 unzip (skip 파라미터를 True로 지정하여 이미 파일이 존재하면 건너뜀)

    load_img_option = {"max_img_per_class" : pc.MAX_IMG_PER_CLASS, "resume" : pc.RESUME, "save" : pc.SAVE}

    if pc.CENTER_CROP : # CENTER_CROP 파리미터가 True인 경우
        load_img_option["center_crop"] = {"crop_width" : pc.IMG_CROP_WIDTH, "crop_height" : pc.IMG_CROP_HEIGHT} # crop 관련 파라미터 추가
    elif pc.EXTRACT_PATCH : # EXTRACT_PATCH 파라미터가 True인 경우 patch 추출 파라미터 추가
        load_img_option["patch_option"] =  {'dim': pc.PATCH_DIM,
        'offset': pc.PATCH_OFFSET,
        'stride': pc.PATCH_STRIDE,
        'rand': pc.PATCH_RAND,
        'function': pc.PATCH_HANDLER,
        'threshold': pc.PATCH_THRESHOLD,
        'num': pc.N_MAX_PATCH,
        'save_interval' : pc.PATCH_SAVE_INTERVAL
        }

    img_data, label = load_img_data(img_dir_path_list, option = load_img_option)  # 이미지 데이터와 레이블 불러옴

    oneLabel = data_processing.label_to_number(label, onehot = True)    # string label을 one hot vector로 변환

    cnn_model = CNN()
    cnn_model.train(img_data, oneLabel , tc.EPOCH, tc.BATCH_SIZE)
    
    # idx = np.random.randint(0, len(img_data), 10)
    # print(cnn_model.predict(np.array(img_data)[idx]))
    # print(np.array(num_label)[idx])
    

    

