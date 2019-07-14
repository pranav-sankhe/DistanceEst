import tensorflow as tf 
num_iterations = 10000
LR = 0.0001		
init_std = 0.1

data_path = "../data/sorted_data/result.csv"

batch_size = 1024
lstm_sizes = [64, 128]
keep_prob_ = 0.75 # fixed

dense_units = num_labels = 30 #1 for regression

fc_size = [64, 128, num_labels]

DTYPE = tf.float32


num_nodes = 5
num_time = 20
log_dir = './train'
save_path = './saved_model/model.ckpt"'

max_gradient_norm = 10
# test_labels_length = 2814

test_bool = False
train_bool = True	

warmup_steps = 0
#How to warmup learning rates. Options include: 
#t2t: Tensor2Tensor's way, start with lr 100 times smaller, then exponentiate until the specified lr.\
warmup_scheme = "t2t" 
