import data_processing
from config import CommonConfig as cc
from config import TrainConfig as tc
from config import PreprocessingConfig as pc

from functools import partial
import os
import numpy as np
import pickle
import tensorflow as tf
from tqdm import tqdm
from PIL import Image


# TODO : 함수 주석 달기 및 모듈화
# HEIC 이미지 파일 변환 -> HEIC 이미지는 사용하지 않음.

def unzip_all_img_file(skip=False):
    dest_dir_path = os.path.join(cc.PROJECT_PATH, cc.IMG_ROOT_DIR_NAME)  # 압축해제 된 이미지를 저장할 디렉토리의 경로

    data_processing.unzip_all([cc.ORIGINAL_PHONE_IMG_DIR_PATH, cc.ORIGINAL_CAMERA_IMG_DIR_PATH], dest_dir_path,
                              skip=skip)


def load_img_data(img_dir_path_list, pickle_file_path,
                  option):  # 이미지가 저정된 디렉토리 하위에 저장된 이미지 데이터를 모두 불러오는 함수 (pillow의 Image 사용)
    # option
    max_img_per_class = option["max_img_per_class"]
    resume = option["resume"]
    save = option["save"]
    img_width = option["crop"]["width"]
    img_height = option["crop"]["height"]

    cur_model_index = 0  # 현재 처리중인 모델의 index
    error_file_name_list = []  # array로 변환하는데 실패한 이미지 파일명 목록
    X, y = [], []

    pickle_dir_path = os.path.dirname(pickle_file_path)  # pickle 파일이 위치한 디렉토리의 path
    pickle_path_except_ext, pickle_ext = os.path.splitext(pickle_file_path)  # pickle 파일의 확장자와 나머지 path를 분리
    pickle_file_path = os.path.join(pickle_dir_path,
                                    f"{pickle_path_except_ext}_{img_width}_{img_height}_{max_img_per_class}{pickle_ext}")  # pickle 파일명을 "pickle 파일명_너비_높이_모델당 이미지 개수.확장자"로 변경

    print(pickle_file_path)

    total_model_dir_path_list = [os.path.join(img_dir_path, img_path) for img_dir_path in img_dir_path_list for img_path
                                 in os.listdir(
            img_dir_path)]  # img_dir_path_list의 각각의 img_dir_path에 있는 파일 및 디렉토리의 path 목록(즉, 카메라 모델 디렉토리의 path 목록)

    if resume == True:  # resume 파라미터가 True인 경우, pickle 파일에서 데이터를 불러와 작업을 재개
        data = data_processing.load_data(pickle_file_path)  # pickle 파일에서 데이터를 불러옴
        if data != None:  # pickle_file_path에 파일이 존재하는 경우
            X = data["X"]  # array로 변환된 이미지 데이터
            y = data["y"]  # label
            cur_model_index = data["save_model_index"] + 1  # 현재 처리중인 모델의 index
            error_file_name_list = data["error_file_name_list"]  # 현재 처리중인 모델의 index

    n_model = len(total_model_dir_path_list)  # 카메라 모델 개수

    with tqdm(total=n_model, initial=cur_model_index, desc="Load Img Data") as model_bar:
        while cur_model_index < n_model:
            processed_model_img_cnt = 0;  # 처리된 현재 카메라 모델의 이미지 개수
            cur_model_dir_path = total_model_dir_path_list[cur_model_index]  # 현재 처리중인 모델 디렉토리의 경로
            all_files_path = data_processing.get_all_files(cur_model_dir_path)  # 카메라 모델 디렉토리 하위의 모든 파일 path
            model_name = os.path.basename(cur_model_dir_path)  # 현재 처리중인 모델의 이름

            for file_path in all_files_path:  # 현재 모델 디렉토리 하위의 모든 파일에 대해 반복
                try:
                    img = Image.open(
                        file_path)  # Image 모듈을 이용하여 이미지 파일을 불러옴 (image를 많이 불러오면 too many open files 에러가 발생하므로 with 구문을 사용)

                    croped_img = data_processing.center_crop(img, img_width,
                                                             img_height)  # TODO : 이미지를 center_crop (일단 이미지 cropping, 후에 네트워크가 완성되면 (saturated pixel/ image dynamic 또는 quality function에 따른) patch priority 적용하는 걸로 변경)
                    X.append(np.asarray(croped_img))  # 이미지를 numpy array로 변환
                    y.append(model_name)
                    processed_model_img_cnt += 1  # 처리된 현재 카메라 모델의 이미지 개수 증가
                except Exception as err:
                    print(err)  # 에러 내용 출력
                    error_file_name_list.append(os.path.basename(file_path))  # 이미지를 불러올 때 error가 발생한 경우 해당 파일명 저장

                if max_img_per_class != None and processed_model_img_cnt >= max_img_per_class:  # 각 모델에 대해 max_img_per_class개의 이미지만 array로 변환
                    break;

            if save == True:  # save 파라미터가 True(default)인 경우 pickle 파일에 저장
                data_processing.save_data(
                    {"X": X, "y": y, "save_model_index": cur_model_index, "error_file_name_list": error_file_name_list},
                    pickle_file_path)  # array로 변환된 이미지 데이터, 레이블, 저장 완료된 카메라 모델의 index를 pickle 파일에 저장

            cur_model_index += 1  # 현재 처리중인 모델의 index 증가
            model_bar.update()  # tqdm update

    print("error_cnt : " + str(len(error_file_name_list)))  # image를 불러와 center crop하고, array로 변환할 때 에러가 발생한 파일의 개수 출력
    print(error_file_name_list)  # 에러가 발생한 파일명 출력

    return X, y  # array로 변환된 이미지 데이터와 레이블 반환


class CNN:
    def __init__(self):
        self.session = tf.Session()  # TODO : 저장할 모델을 불러올 수 있도록 수정
        self.build_net()

    def build_net(self):
        self.X = tf.placeholder(tf.float32, [None, 64, 64, 3])
        self.y = tf.placeholder(tf.int32, [None])

        # TODO : 이미지를 numpy로 변환하지 않고, 이미지 그대로를 network에 입력하는 방법은?
        # TODO : 이미지에서 64 * 64 크기의 패치 k를 추출
        # TODO : saturated 픽셀이 있는 패치 제외
        # TODO : 패치에 우선순위를 지정 (평균 값이 image dynamic의 절반에 가까운지에 따라)
        # TODO : 각 패치에 전체 이미지와 동일한 레이블을 지정
        # TODO : 각 입력 patch에서 훈련 세트에 대한 픽셀 단위 평균을 빼줌.
        # TODO : dynamic을 줄이기 위해 진폭을 0.0125배로 조정
        # TODO : 초기화 함수 지정

        custom_conv2d = partial(tf.layers.conv2d, activation=tc.CONV_ACTIVATION,
                                kernel_initializer=tc.CONV_KERNEL_INITIALIZER,
                                bias_initializer=tc.CONV_BIAS_INITIALIZER)
        custom_max_pooling2d = partial(tf.layers.max_pooling2d, pool_size=tc.MAX_POOLING_POOL_SIZE,
                                       strides=tc.MAX_POOLING_STRIDES, padding=tc.MAX_POLLING_PADDING)
        custom_dense = partial(tf.layers.dense, units=tc.UNITS, activation=tc.ACTIVATION_FUNC)

        l1_conv = custom_conv2d(inputs=self.X, filters=tc.L1_CONV_FILTERS, kernel_size=tc.L1_CONV_KERNEL_SIZE,
                                padding=tc.CONV_PADDING)
        l1_max_pool = custom_max_pooling2d(l1_conv)

        l2_conv = custom_conv2d(inputs=l1_max_pool, filters=tc.L2_CONV_FILTERS, kernel_size=tc.L2_CONV_KERNEL_SIZE,
                                padding=tc.CONV_PADDING)
        l2_max_pool = custom_max_pooling2d(l2_conv)

        l3_conv = custom_conv2d(inputs=l2_max_pool, filters=tc.L3_CONV_FILTERS, kernel_size=tc.L3_CONV_KERNEL_SIZE,
                                padding=tc.CONV_PADDING)
        l3_max_pool = custom_max_pooling2d(l3_conv)

        l4_conv = custom_conv2d(inputs=l3_max_pool, filters=tc.L4_CONV_FILTERS, kernel_size=tc.L4_CONV_KERNEL_SIZE,
                                padding=tc.CONV_PADDING)

        l4_flatten = tf.layers.flatten(l4_conv)
        self.ip1 = custom_dense(inputs=l4_flatten)

        self.logits = custom_dense(inputs=self.ip1)
        self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y, logits=self.logits))

        self.optimizer = tf.train.MomentumOptimizer(learning_rate=tc.BASE_LR,
                                                    momentum=tc.MOMEMTUM)  # TODO : leaning_schedule 지정, weight_decay 적용할 수 있도록 optimizer 변경

        self.training_op = self.optimizer.minimize(self.loss)
        self.predicted = tf.argmax(self.logits, 1)  # TODO : what is parameter 1???

    def train(self, X_data, y_data, n_epoch, batch_size):
        np.random.seed(tc.BATCH_RANDOM_SEED)  # randint SEED 지정

        n_examples = len(X_data)  # 이미지의 개수
        n_iteration = int(n_examples / batch_size)  # iteration = 이미지 개수 / batch size

        self.session.run(tf.global_variables_initializer())  # 변수 초기화

        model_file_dir_path = os.path.join(cc.PROJECT_PATH,
                                           f"{tc.MODEL_FILE_NAME}_{pc.IMG_WIDTH}_{pc.IMG_HEIGHT}_{pc.MAX_IMG_PER_CLASS}")  # 모델을 저장할 디렉토리의 경로
        model_file_path = os.path.join(model_file_dir_path, tc.MODEL_FILE_NAME)  # 모델을 저장할 파일의 경로
        log_dir_path = os.path.join(model_file_dir_path, tc.LOG_DIR_NAME)

        data_processing.create_empty_dir(model_file_dir_path)  # 모델 파일을 저장할 디렉토리가 없다면 생성, 이미 있다면 디렉토리를 비움

        loss_sum = 0  # 배치당 정확도를 모두 더한 값을 저장

        saver = tf.train.Saver()  # 학습된 모델을 저장할 Savar 객체
        file_writer = tf.summary.FileWriter(log_dir_path, tf.compat.v1.get_default_graph())

        for cur_epoch in tqdm(range(n_epoch), desc="Epoch"):
            s_loss = 0  # epoch의 배치당 loss 총합

            for cur_iteration in range(n_iteration):
                shuffled_indices = np.random.randint(0, n_examples,
                                                     batch_size)  # 0 ~ (이미지 데이터 개수 -1) 사이의 수를 batch size만큼 random 추출

                # 모든 batch의 loss / 배치 개수로 loss 계산
                X_batch = np.array(X_data)[shuffled_indices]  # random image data batch
                y_batch = np.array(y_data)[shuffled_indices]  # random label batch

                ip1, cur_loss, _ = self.session.run([self.ip1, self.loss, self.training_op],
                                                    feed_dict={self.X: X_batch, self.y: y_batch})

                """ TODO : tensorboard 코드 수정 
                if cur_iteration % tc.LOG_WRITE_INTERVAL == 0 : 
                    steps = cur_epoch * n_iteration * cur_iteration
                    file_writer.add_summary(cur_loss, steps)
                """

                print(cur_loss)
                # s_loss += cur_loss

            # print("Epoch : ", cur_epoch, "Cur Epoch Avg loss : ", s_loss  / n_iteration)  # epoch index와 해당 epoch의 평균 loss 출력

            if cur_epoch % tc.MODEL_SAVE_INTERVAL == 0:  # MODEL_SAVE_INTERVAL epoch마다 모델을 save
                saver.save(self.session, model_file_path)  # 모델 저장

        saver.save(self.session, model_file_path)  # 모델 저장

    def extract_feature_vectors(self, X_data):
        return self.session.run(self.ip1, feed_dict={self.X: X_data})  # TODO : relu func을 적용한 벡터를 svm에 입력하는 것인지 확인

    def predict(self, X_data):
        return self.session.run(self.predicted, feed_dict={self.X: X_data})


# TODO : Galaxy Note9 SM-N960N는 제외한 후 실험
# TODO : Train/Test set 분할 (모델마다 섞은 후 20퍼센트)

# tqdm 라이브러리로 반복문 진행율 찍어볼 수 있음
# 이미지 저장 시에는 .npy 또는 tensorflow의 record 사용
# 학습 시, keras의 generator 또는 tensorflow의 dictionary 사용
# 첫 에폭 로스가 대략 log2(클래스 개수)정도면 나쁜 네트워크는 아님

if __name__ == "__main__":
    img_data_pickle_path = os.path.join(cc.PROJECT_PATH, cc.IMG_DATA_PICKLE_NAME)  # pickle 파일 path

    img_dir_path_list = [cc.PHONE_IMG_DIR_UNZIP_PATH]  # 스마트폰 이미지만 사용

    # unzip_all_img_file(skip = True) # 이미지 파일을 모두 unzip (skip 파라미터를 True로 지정하여 이미 파일이 존재하면 건너뜀)

    img_data, label = load_img_data(img_dir_path_list, img_data_pickle_path,
                                    option={"max_img_per_class": pc.MAX_IMG_PER_CLASS,
                                            "resume": pc.RESUME, "save": pc.SAVE,
                                            "crop": {"width": pc.IMG_WIDTH,
                                                     "height": pc.IMG_HEIGHT}})  # 이미지 데이터와 레이블 불러옴

    num_label = data_processing.label_to_number(label)  # string label을 숫자형으로 변환

    cnn_model = CNN()
    cnn_model.train(img_data, num_label, tc.EPOCH, tc.BATCH_SIZE)
    idx = np.random.randint(0, len(img_data), 10)

    print(cnn_model.predict(np.array(img_data)[idx]))
    print(np.array(num_label)[idx])

