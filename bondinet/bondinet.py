from data_processing import create_dir

from functools import partial
import os
from tqdm import tqdm
import math
import numpy as np
import tensorflow as tf
import itertools
import re

class Bondinet : # https://minimin2.tistory.com/36
    MODEL_FILE_NOT_EXIST_MSG = "Model File is not exists"
    TOO_BIG_BATCH_SIZE = "of example is smaller than batch size!"

    def __init__(self, 
                 n_classes, 
                 weight_decay) : 
        
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))) #  https://newsight.tistory.com/255 
        
        self.N_CLASSES = n_classes
        self.WEIGHT_DECAY = weight_decay
        self.IP2_UNITS = n_classes

    def build_net(self, optimizer) :
        # conv common parameter
        CONV_PADDING = "valid"
        CONV_ACTIVATION = None
        CONV_KERNEL_INITIALIZER = "glorot_uniform"

        CONV_BIAS_INITIALIZER = None
        USE_CONV_BIAS = False

        # l1 conv parameter
        L1_CONV_FILTERS = 32
        L1_CONV_KERNEL_SIZE = [4, 4]

        # l2 conv parameter
        L2_CONV_FILTERS = 48
        L2_CONV_KERNEL_SIZE = [5, 5]

        # l3 conv parameter
        L3_CONV_FILTERS = 64
        L3_CONV_KERNEL_SIZE = [5, 5]

        # l4 conv parameter
        L4_CONV_FILTERS = 128
        L4_CONV_KERNEL_SIZE = [5, 5]

        # max_pooling parameter
        MAX_POOLING_POOL_SIZE = [2, 2]
        MAX_POOLING_STRIDES = [2, 2]
        MAX_POLLING_PADDING = "same"    

        # ip1 layer parameter
        IP1_UNITS = 128
        IP1_ACTIVATION_FUNC = tf.nn.relu

        # ip2 layer parameter
        IP2_ACTIVATION_FUNC = None

        def weight_summary(weight) : # weight의 summary 생성 (http://solarisailab.com/archives/710)
            weight_name = weight.name.split(":")[0]
            with tf.name_scope(weight_name) :
                with tf.name_scope("mean") :
                    mean = tf.reduce_mean(weight)
                    tf.summary.scalar("mean", mean)
                
                with tf.name_scope("stddev") :
                    stddev = tf.sqrt(tf.reduce_mean(tf.square(weight - mean)))
                    tf.summary.scalar("stddev", stddev)

                tf.summary.scalar("max", tf.reduce_max(weight))
                tf.summary.scalar("min", tf.reduce_min(weight))

            with tf.name_scope("Variables/" + weight_name)  : 
                tf.summary.histogram("histogram", weight)

        with tf.name_scope('input'):
            self.X = tf.placeholder(tf.float32, [None, 64, 64, 3])  # None은 Batch size
            self.y = tf.placeholder(tf.int32, [None, self.N_CLASSES])   # OneHot Vector
        
        l2_regularizer = tf.contrib.layers.l2_regularizer(self.WEIGHT_DECAY)
        custom_conv2d = partial(tf.layers.conv2d, activation = CONV_ACTIVATION, kernel_initializer = CONV_KERNEL_INITIALIZER, bias_initializer = CONV_BIAS_INITIALIZER, use_bias = USE_CONV_BIAS, padding = CONV_PADDING, kernel_regularizer = l2_regularizer)
        custom_max_pooling2d = partial(tf.layers.max_pooling2d, pool_size = MAX_POOLING_POOL_SIZE, strides = MAX_POOLING_STRIDES, padding = MAX_POLLING_PADDING)
        custom_dense = partial(tf.layers.dense, kernel_regularizer = l2_regularizer)

        l1_conv = custom_conv2d(inputs = self.X, filters = L1_CONV_FILTERS, kernel_size = L1_CONV_KERNEL_SIZE)
        l1_max_pool = custom_max_pooling2d(l1_conv)
    
        l2_conv = custom_conv2d(inputs = l1_max_pool, filters = L2_CONV_FILTERS, kernel_size = L2_CONV_KERNEL_SIZE)
        l2_max_pool = custom_max_pooling2d(l2_conv)
    
        l3_conv = custom_conv2d(inputs = l2_max_pool, filters = L3_CONV_FILTERS, kernel_size = L3_CONV_KERNEL_SIZE)
        l3_max_pool = custom_max_pooling2d(l3_conv)

        l4_conv = custom_conv2d(inputs = l3_max_pool, filters = L4_CONV_FILTERS, kernel_size = L4_CONV_KERNEL_SIZE)

        l4_flatten = tf.layers.flatten(l4_conv) # TODO : 논문 코드에는 flatten 과정이 없는데, IP1 layer에 flatten하여 입력하는 것이 맞는지 확인
        self.ip1 = custom_dense(inputs = l4_flatten, units = IP1_UNITS, activation = IP1_ACTIVATION_FUNC)
        
        self.logits = custom_dense(inputs = self.ip1, units = self.IP2_UNITS, activation = IP2_ACTIVATION_FUNC)
        self.base_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = self.y, logits = self.logits)) # https://stats.stackexchange.com/questions/327348/how-is-softmax-cross-entropy-with-logits-different-from-softmax-cross-entropy-wi
        
        regularization_losses = tf.losses.get_regularization_losses()
        self.reg_loss = tf.add_n([self.base_loss] + regularization_losses)    # l2 regulariztion loss (https://stackoverflow.com/questions/37107223/how-to-add-regularizations-in-tensorflow, http://blog.naver.com/PostView.nhn?blogId=infoefficien&logNo=221143520556)
        self.loss_summary = tf.summary.scalar("loss", self.reg_loss)
        
        gradients = optimizer.compute_gradients(loss = self.reg_loss)

        l2_norm = lambda t: tf.sqrt(tf.reduce_sum(tf.pow(t, 2)))    
        for gradient, variable in gradients:    # https://matpalm.com/blog/viz_gradient_norms/
          tf.summary.histogram("Gradients/" + variable.name, l2_norm(gradient)) # 각 weight에 대한 gradient의 summary 생성
          weight_summary(variable) # 각 가중치의 summary 생성

        self.training_op = optimizer.apply_gradients(gradients)

        self.predicted = tf.argmax(self.logits, 1)    # what is parameter 1? - axis
        self.y_true = tf.argmax(self.y, 1)
        is_correct = tf.equal(self.predicted, self.y_true)
        self.accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32)) # accuracy 계산 (tf.cast : Casts a tensor to a new type. (The type of is_correct is int64))

        self.accuracy_summary = tf.summary.scalar("accuracy", self.accuracy)
        self.loss_accuracy_summary = tf.summary.merge([self.loss_summary, self.accuracy_summary])
        self.merged = tf.summary.merge_all()

    def train(self, 
              X_train, 
              y_train, 
              X_val,
              y_val,
              n_epoch, 
              batch_size, 
              optimizer,
              momentum,
              base_lr,
              lr_scheduler,
              lr_decay_interval,
              lr_decay_rate,
              model_dir_path,  # 모델을 저장할 디렉토리의 경로
              batch_random_seed = 42, 
              model_save_interval = 10,
              log_write_interval = 10, resume = False) :
        
        LOG_DIR_NAME = "logs"
        MODEL_FILE_NAME = "model.ckpt"

        np.random.seed(batch_random_seed)    # randint SEED 지정

        n_train_examples = len(X_train)  # Train 이미지의 개수
        n_val_examples = len(X_val) # Validation 이미지 개수
        n_iteration = int(n_train_examples / batch_size) # epoch당 iteration 개수 = 이미지 개수 / batch size
        
        if n_train_examples < batch_size : # batch size 보다 샘플의 개수가 적은 경우
            raise Exception(self.TOO_BIG_BATCH_SIZE)

        if lr_scheduler : # lr_scheduler를 사용하는 경우
            if optimizer == "Adam" : 
                raise Exception("A learning scheduler is specified in the adam optimizer.") # 실험을 위해 optimizer가 SGD일 때만 learning_scheduler를 지정할 수 있도록 함.
            global_step = tf.Variable(0, trainable=False)
            decay_steps = lr_decay_interval * n_iteration  
            learning_rate = tf.train.exponential_decay(learning_rate = base_lr, global_step = global_step, decay_steps = decay_steps, decay_rate = lr_decay_rate, staircase = True)
        else : # lr_scheduler를 사용하지 않는 경우
            if optimizer == "SGD":
                raise Exception("A learning scheduler is not specified in the SGD optimizer.") # 실험을 위해 SGD일 때는 learning_scheduler를 지정하도록 함.
            if lr_decay_interval is not None : 
                raise Exception("If you don't specify a learning scheduler, lr_decay_interval is not required.")
            if lr_decay_rate is not None : 
                raise Exception("If you don't specify a learning scheduler, lr_decay_rate is not required.")
            learning_rate = base_lr

        if optimizer == "Adam" : 
            optimizer = tf.train.AdamOptimizer(learning_rate= learning_rate, beta1 = momentum)  # TODO : leaning_schedule 지정, weight_decay 적용할 수 있도록 optimizer 변경, sgd로 변경
        elif optimizer == "SGD":
            optimizer = tf.train.MomentumOptimizer(learning_rate= learning_rate, momentum = momentum)  # TODO : leaning_schedule 지정, weight_decay 적용할 수 있도록 optimizer 변경, sgd로 변경
        else : # Adam과 SGD를 제외한 optimizer인 경우
            raise Exception("Unsupported optimizer".format(optimizer))

        self.build_net(optimizer)   # 네트워크 build

        log_dir_path = os.path.join(model_dir_path, LOG_DIR_NAME)
        train_log_dir_path = os.path.join(log_dir_path, "Train")    # Train 이벤트 파일 저장 디렉토리
        val_log_dir_path = os.path.join(log_dir_path, "Val")    # Validation 이벤트 파일 저장 디렉토리

        create_dir([model_dir_path, log_dir_path, train_log_dir_path, val_log_dir_path])  # log와 model를 저장할 디렉터리가 없다면 생성

        model_dir_list = sorted(list(map(int, [dir_name for dir_name in os.listdir(model_dir_path) if re.search("^[0-9]*$", dir_name)])), reverse = True) # 디렉토리명이 숫자인 디렉토리명만 추출한 후, int로 형변환 한 다음, 정렬
        
        max_to_keep = int(math.ceil(n_epoch / model_save_interval)) # Checkpoint의 최대 개수 (interval에 따라 모든 모델을 저장)
        saver = tf.train.Saver(max_to_keep  = max_to_keep)    # 학습된 모델을 저장할 Savar 객체

        if resume :
            if model_dir_list : # 모델 디렉토리가 있는 경우
                saved_epoch = model_dir_list[0] # 가장 마지막에 저장된 epoch
                start_epoch = saved_epoch + 1 # 이전에 저장된 가장 마지막 epoch 다음부터 학습 재개
                resume_model_file_path = os.path.join(model_dir_path, str(saved_epoch), MODEL_FILE_NAME)    # 가장 마지막 epoch를 저장한 model 파일

                saver.restore(self.sess, save_path = resume_model_file_path)    # 학습 재개를 위해 variable 복원
            else :  # 모델 디렉토리가 없는 경우
                raise Exception(self.MODEL_FILE_NOT_EXIST_MSG)   
        else : 
            self.sess.run(tf.global_variables_initializer())     # 변수 초기화
            start_epoch = 0

        
        train_writer = tf.summary.FileWriter(train_log_dir_path, self.sess.graph)   # saver (http://solarisailab.com/archives/710 참조 )
        val_writer = tf.summary.FileWriter(val_log_dir_path, self.sess.graph)   # val, train summary (https://gist.github.com/chang12/2d52af6a191a3aa4250cf44926dfd48a, https://stackoverflow.com/questions/37146614/tensorboard-plot-training-and-validation-losses-on-the-same-graph 참조)

        cur_epoch = start_epoch
        with tqdm(total = n_epoch, initial = cur_epoch, desc = "Epoch") as train_tqdm : 
            while cur_epoch < n_epoch : 
                cur_model_dir_path = os.path.join(model_dir_path, str(cur_epoch))   # 현재 epoch에 해당하는 모델을 저장할 디렉토리 path
                cur_model_file_path = os.path.join(cur_model_dir_path, MODEL_FILE_NAME) # 현재 epoch에 해당하는 모델을 저장할 file path
            
                for cur_iteration in tqdm(range(n_iteration), desc = "Iteration") :  
                    steps = cur_epoch * n_iteration + cur_iteration
                    train_shuffled_indices = np.random.randint(0, n_train_examples, batch_size) # 훈련 데이터에서 0 ~ (이미지 데이터 개수 -1) 사이의 수를 batch size만큼 random 추출

                    # 모든 batch의 loss / 배치 개수로 loss 계산
                    X_train_batch = np.array(X_train)[train_shuffled_indices]  # random image data batch
                    y_train_batch = np.array(y_train)[train_shuffled_indices]  # random label batch
                
                    train_feed_dict = {self.X : X_train_batch / 255, self.y : y_train_batch}
                    
                    if steps % log_write_interval == 0 : # log를 기록하는 iteration인 경우만 summary 계산
                        val_shuffled_indices = np.random.randint(0, n_val_examples, batch_size) # validation set에서 batch size만큼 random 추출
                        
                        X_val_batch = np.array(X_val)[val_shuffled_indices]  # validation X batch
                        y_val_batch = np.array(y_val)[val_shuffled_indices]  # random label batch

                        train_summary, _ = self.sess.run([self.merged, self.training_op], feed_dict = train_feed_dict)   
                        val_loss_accuracy_summary = self.sess.run(self.loss_accuracy_summary, feed_dict = {self.X : X_val_batch / 255, self.y : y_val_batch})    # val accuracy, loss 계산

                        train_writer.add_summary(train_summary , steps) # train summary 기록
                        val_writer.add_summary(val_loss_accuracy_summary, steps)    # val accuracy, loss 기록
                        train_writer.flush()
                        val_writer.flush()
                    else : 
                        self.sess.run(self.training_op, feed_dict = train_feed_dict)   

                if cur_epoch % model_save_interval == 0 or cur_epoch == n_epoch - 1:    # 마지막 epoch이거나, MODEL_SAVE_INTERVAL epoch마다 모델을 save
                    if not os.path.exists(cur_model_dir_path) : # 현재 epoch에 해당하는 model을 저장할 디렉터리가 없다면 생성
                        os.makedirs(cur_model_dir_path)

                    saver.save(self.sess, cur_model_file_path)   # 모델 저장
                
                train_tqdm.update()
                cur_epoch += 1

    def extract_feature_vectors(self, X_data) : # relu layer에서 forward propagation을 stop하여 feature 벡터 추출
        return self.sess.run(self.ip1, feed_dict = {self.X : X_data})
         
    def predict(self, X_data) :
        return self.sess.run(self.predicted, feed_dict = {self.X : X_data})

    def plot_confusion_matrix(cm, class_names):
        """
        Returns a matplotlib figure containing the plotted confusion matrix.
    
        Args:
           cm (array, shape = [n, n]): a confusion matrix of integer classes
           class_names (array, shape = [n]): String names of the integer classes
        """
    
        figure = plt.figure(figsize=(8, 8))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title("Confusion matrix")
        plt.colorbar()
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45)
        plt.yticks(tick_marks, class_names)
    
        # Normalize the confusion matrix.
        cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)
    
        # Use white text if squares are dark; otherwise black.
        threshold = cm.max() / 2.
    
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            color = "white" if cm[i, j] > threshold else "black"
            plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)
        
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        return figure