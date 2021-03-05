import data_processing

from functools import partial
import os
from tqdm import tqdm
import math
import numpy as np
import tensorflow as tf
import sklearn
import itertools

class Bondinet : # https://minimin2.tistory.com/36
    def __init__(self, 
                 n_classes, 
                 base_lr, 
                 weight_decay, 
                 momentum, 
                 lr_decay_interval, 
                 lr_decay_rate) : 
        
        self.session = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))) #  https://newsight.tistory.com/255 
        
        self.N_CLASSES = n_classes
        self.BASE_LR = base_lr
        self.WEIGHT_DECAY = weight_decay
        self.MOMENTUM = momentum
        self.LR_DECAY_INTERVAL = lr_decay_interval
        self.LR_DECAY_RATE = lr_decay_rate
        self.IP2_UNITS = n_classes

    def build_net(self, decay_steps) :
        # conv common parameter
        CONV_PADDING = "same"
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

        # LEARNING_SCHEDULE_NAME = "learning_schedule"
        
        def weight_summary(weight) : # weight의 summary 생성 (http://solarisailab.com/archives/710)
            weight_name = weight.name.split(":")[0]
            with tf.name_scope(f"{weight_name}") :
                with tf.name_scope("mean") :
                    mean = tf.reduce_mean(weight)
                    tf.summary.scalar("mean", mean)
                
                with tf.name_scope("stddev") :
                    stddev = tf.sqrt(tf.reduce_mean(tf.square(weight - mean)))
                    tf.summary.scalar("stddev", stddev)

                tf.summary.scalar("max", tf.reduce_max(weight))
                tf.summary.scalar("min", tf.reduce_min(weight))
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
        tf.summary.scalar("loss", self.reg_loss)
        
        global_step = global_step = tf.Variable(0, trainable=False)
        # learning_schedule = tf.train.exponential_decay(learning_rate = self.BASE_LR, global_step = global_step, decay_steps = decay_steps, decay_rate = self.LR_DECAY_RATE, staircase = True, name = self.LEARNING_SCHEDULE_NAME)

        self.optimizer = tf.train.AdamOptimizer(learning_rate= self.BASE_LR, beta1 = self.MOMENTUM)  # TODO : leaning_schedule 지정, weight_decay 적용할 수 있도록 optimizer 변경, sgd로 변경
        gradients = self.optimizer.compute_gradients(loss = self.reg_loss)

        l2_norm = lambda t: tf.sqrt(tf.reduce_sum(tf.pow(t, 2)))    
        for gradient, variable in gradients:    # https://matpalm.com/blog/viz_gradient_norms/
          tf.summary.histogram("gradients/" + variable.name, gradient) # 각 weight에 대한 gradient의 summary 생성
          weight_summary(variable) # 각 가중치의 summary 생성

        self.training_op = self.optimizer.apply_gradients(gradients)

        self.predicted = tf.argmax(self.logits, 1)    # what is parameter 1? - axis
        self.y_true = tf.argmax(self.y, 1)
        is_correct = tf.equal(self.predicted, self.y_true)
        self.accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32)) # accuracy 계산 (tf.cast : Casts a tensor to a new type. (The type of is_correct is int64))

        tf.summary.scalar("accuracy", self.accuracy)
        self.merged = tf.summary.merge_all()

    def train(self, 
              X_data, 
              y_data, 
              n_epoch, 
              batch_size, 
              model_dir_path,  # 모델을 저장할 디렉토리의 경로
              batch_random_seed = 42, 
              model_save_interval = 10,
              log_write_interval = 10) :

        LOG_DIR_NAME = "logs"
        MODEL_FILE_NAME = "model"

        np.random.seed(batch_random_seed)    # randint SEED 지정

        n_examples = len(X_data)  # 이미지의 개수
        n_iteration = int(n_examples / batch_size) # iteration = 이미지 개수 / batch size
        
        if n_examples < batch_size : # batch size 보다 샘플의 개수가 적은 경우
            raise Exception("# of example is smaller than batch size!")

        decay_steps = self.LR_DECAY_INTERVAL * n_iteration
        self.build_net(decay_steps)
        
        self.session.run(tf.global_variables_initializer()) # 변수 초기화
        
        """
        if use_learning_rate_decay : 
            learning_rate = self.session.graph.get_tensor_by_name(f"{self.learning_schedule_name}:0")
            print(tf.train.exponential_decay(learning_rate  = self.base_lr, global_step = self.global_step, decay_steps = self.lr_decay_interval * n_iteration, decay_rate = self.lr_decay_rate, staircase = True))
            tf.assign(learning_rate, tf.train.exponential_decay(learning_rate = self.base_lr, global_step = self.global_step, decay_steps = self.lr_decay_interval * n_iteration, decay_rate = self.lr_decay_rate, staircase = True))  # TODO : learning_rate를 base로 지정해줘도 괜찮을지 생각.        
        """

        log_dir_path = os.path.join(model_dir_path, LOG_DIR_NAME)

        if not os.path.exists(log_dir_path): # log를 저장할 디렉터리가 없다면 생성
            os.makedirs(log_dir_path)
        if not os.path.exists(model_dir_path) : # model을 저장할 디렉터리가 없다면 생성
            os.makedirs(model_dir_path)

        max_to_keep = int(math.ceil(n_epoch / model_save_interval)) # Checkpoint의 최대 개수 (interval에 따라 모든 모델을 저장)

        saver = tf.train.Saver(max_to_keep  = max_to_keep)    # 학습된 모델을 저장할 Savar 객체
        file_writer = tf.summary.FileWriter(log_dir_path, self.session.graph)   # saver http://solarisailab.com/archives/710 참조 

        for cur_epoch in tqdm(range(n_epoch), desc = "Epoch") :  
            cur_model_dir_path = os.path.join(model_dir_path, str(cur_epoch))
            cur_model_file_path = os.path.join(cur_model_dir_path, MODEL_FILE_NAME)
            
            if not os.path.exists(cur_model_dir_path) : # 현재 epoch에 해당하는 model을 저장할 디렉터리가 없다면 생성
                os.makedirs(cur_model_dir_path)

            for cur_iteration in tqdm(range(n_iteration), desc = "Iteration") :  
                steps = cur_epoch * n_iteration + cur_iteration
                shuffled_indices = np.random.randint(0, n_examples, batch_size) # 0 ~ (이미지 데이터 개수 -1) 사이의 수를 batch size만큼 random 추출

                # 모든 batch의 loss / 배치 개수로 loss 계산
                X_batch = np.array(X_data)[shuffled_indices]  # random image data batch
                y_batch = np.array(y_data)[shuffled_indices]  # random label batch
                
                feed_dict = feed_dict = {self.X : X_batch / 255, self.y : y_batch}
                
                if steps % log_write_interval == 0 : # log를 기록하는 iteration인 경우만 summary 계산
                    summary, _ = self.session.run([self.merged, self.training_op], feed_dict = feed_dict)   
                    file_writer.add_summary(summary , steps)
                else : 
                    self.session.run([self.training_op], feed_dict = feed_dict)   

            if cur_epoch % model_save_interval == 0 :    # MODEL_SAVE_INTERVAL epoch마다 모델을 save
                saver.save(self.session, cur_model_file_path)   # 모델 저장
            
        saver.save(self.session, cur_model_file_path) # 모델 저장

    def extract_feature_vectors(self, X_data) : # relu layer에서 forward propagation을 stop하여 feature 벡터 추출
        return self.session.run(self.ip1, feed_dict = {self.X : X_data})
         
    def predict(self, X_data) :
        return self.session.run(self.predicted, feed_dict = {self.X : X_data})

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