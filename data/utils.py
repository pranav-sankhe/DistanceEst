from __future__ import division 
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler 
import tensorflow as tf
import pandas as pd 
import numpy as np 
from sklearn.externals import joblib
import matplotlib.pyplot as plt
import multiprocessing as mp
import hparams

REF_ETA_DIST = 191.08
REF_ETA_RSSI = 65

ROUTER_POS = [0,137.6]
X_STATIC_NODES = [228,228,-186.8,-186.8]
Y_STATIC_NODES = [-147.6,137.6,-147.6,137.6]



def normalize_data(X):
    mean = np.mean(X)
    variance = np.var(X)

    norm_X = X - mean
    norm_X = norm_X/np.sqrt(variance)

    return norm_X

def normalize_dist(dist):
	mean = np.mean(dist)
	variance = np.var(dist)
	norm_dist = dist - mean
	norm_dist = norm_dist/np.sqrt(variance)
	return norm_dist


def _weight_variable(name, shape):
    return tf.get_variable(name, shape, hparams.DTYPE, tf.truncated_normal_initializer(stddev=hparams.init_std)) # function to intilaize weights for each layer

def _bias_variable(name, shape):
    return tf.get_variable(name, shape, hparams.DTYPE, tf.constant_initializer(0.1, dtype=tf.float32)) # fucntion to intiliaze bias vector for each layer


def conv_layer(prev_layer, in_filters, out_filters, Ksize, stride, poolTrue, name_scope):

    
    with tf.variable_scope(name_scope) as scope: # name of the block  
        
        kernel = _weight_variable('weights', [Ksize, Ksize, in_filters, out_filters]) # (kernels = filters as defined in TF doc). kernel size = 5 (5*5*5) 
        conv = tf.nn.conv3d(prev_layer, kernel, [stride, stride, stride, stride], padding='SAME')
        biases = _bias_variable('biases', [out_filters])
        bias = tf.nn.bias_add(conv, biases) # define biases for conv1 
        conv1 = tf.nn.relu(bias, name=scope.name) # define the activation for conv1 
        prev_layer = conv1                                                                              
    if poolTrue:    
        pool1 = tf.nn.max_pool3d(prev_layer, ksize=[1, 3, 3, 3, 1], strides=[1, 2, 2, 2, 1], padding='SAME')        
        norm1 = pool1  # tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta = 0.75, name='norm1')
        layer = norm1

    return layer    


def fully_connected(size, prev_layer, name_scope):
    with tf.variable_scope(name_scope) as scope:
        dim = np.prod(prev_layer.get_shape().as_list()[1:])
        prev_layer_flat = tf.reshape(prev_layer, [-1, dim])
        weights = _weight_variable('weights', [dim, size])
        biases = _bias_variable('biases', [size])
        output = tf.nn.relu(tf.matmul(prev_layer_flat, weights) + biases, name=scope.name)
    return output




def batch_norm(inputs, training):
    return tf.layers.batch_normalization(
          inputs=inputs, axis=4,
          momentum=hparams._BATCH_NORM_DECAY, epsilon=hparams._BATCH_NORM_EPSILON, center=True,
          scale=True, training=training, fused=True)


# def _get_learning_rate_warmup(global_step, lr):
#     """Get learning rate warmup."""
#     warmup_steps = hparams.warmup_steps
#     warmup_scheme = hparams.warmup_scheme
#     # utils.print_out("  learning_rate=%g, warmup_steps=%d, warmup_scheme=%s" %
#     #                 (hparams.learning_rate, warmup_steps, warmup_scheme))

#     # Apply inverse decay if global steps less than warmup steps.
#     # Inspired by https://arxiv.org/pdf/1706.03762.pdf (Section 5.3)
#     # When step < warmup_steps,
#     #   learing_rate *= warmup_factor ** (warmup_steps - step)
#     if warmup_scheme == "t2t":
#       # 0.01^(1/warmup_steps): we start with a lr, 100 times smaller
#         warmup_factor = tf.exp(tf.log(0.01) / warmup_steps)
#         inv_decay = warmup_factor**(tf.to_float(warmup_steps - global_step))
#     else:
#         raise ValueError("Unknown warmup scheme %s" % warmup_scheme)

#     return tf.cond(
#         global_step < hparams.warmup_steps,
#         lambda: inv_decay * lr,
#         lambda: lr,
#         name="learning_rate_warump_cond")

# def _get_learning_rate_decay(self, hparams):
#     """Get learning rate decay."""
#     start_decay_step, decay_steps, decay_factor = self._get_decay_info(hparams)
#     utils.print_out("  decay_scheme=%s, start_decay_step=%d, decay_steps %d, "
#                     "decay_factor %g" % (hparams.decay_scheme,
#                                          start_decay_step,
#                                          decay_steps,
#                                          decay_factor))

#     return tf.cond(
#         self.global_step < start_decay_step,
#         lambda: self.learning_rate,
#         lambda: tf.train.exponential_decay(
#             self.learning_rate,
#             (self.global_step - start_decay_step),
#             decay_steps, decay_factor, staircase=True),
#         name="learning_rate_decay_cond")

def compute_eta(rssi_list, Id):
    '''
    rssi_list : array of rssi values of one node
    (x_router, y_router) is the position of the router placed during the measurement. 
    x_static and y_static are the x and y positions of the static nodes during the measurement
    Id is the Id of the node for which we are calculating the ETA  
    '''
    global REF_ETA_RSSI, REF_ETA_DIST
    print 'computing eta for static node', Id
    rssi_list = list(rssi_list)
    
    dist = np.sqrt(np.square(ROUTER_POS[0] - X_STATIC_NODES[Id-1]) + np.square(ROUTER_POS[1] - Y_STATIC_NODES[Id-1]))
    length = len(rssi_list)
    dist = [dist]*length
    
    ref_eta_rssi = np.zeros(length)
    ref_eta_dist = np.zeros(length)
    ref_eta_rssi.fill(REF_ETA_RSSI)
    ref_eta_dist.fill(REF_ETA_DIST)

    rssi_list    = np.array(rssi_list)
    
    num = np.subtract(rssi_list, REF_ETA_RSSI)
    den = (10*np.log10(np.divide(dist, REF_ETA_DIST)))
    eta = np.divide(num, den)
    
    return eta


def dist_from_router(x_list, y_list):
    length = len(x_list)
    x_router = np.array(ROUTER_POS[0])*length
    y_router = np.array(ROUTER_POS[1])*length
    x_pos = np.array(x_list)
    y_pos = np.array(y_list) 
    dist = np.sqrt(np.square(x_pos-x_router) + np.square(y_pos-y_router))

    return dist

def RSSI_id(dataframe,Nodemcu_id):
    object_ids = ['obj0','obj1','obj2','obj3','obj4']
    rssi_columns = ['rssi0','rssi1','rssi2','rssi3','rssi4']
    rssi_list = []
    RSSI_columns = ['obj0','rssi0','obj1','rssi1','obj2','rssi2','obj3','rssi3','obj4','rssi4']
     
    
    for i in range(len(dataframe)): 
        for j in range(5):              
            if len(dataframe[RSSI_columns].loc[i:i].values) > 0:    
                if dataframe[RSSI_columns].loc[i:i][object_ids[j]].values == Nodemcu_id:
                     print "At row  ", i ,' for reference point', Nodemcu_id, 'sorting rssi values'
                     rssi_list = rssi_list + list(dataframe[RSSI_columns].loc[i:i][RSSI_columns[j]].values)
        
    return rssi_list    

def get_dist_data(input):
    Datafile = input['Datafile']                    #input the datafile
    data = pd.read_csv(Datafile)                    #load the datafile and create a dataframe               
    dist = data['distance'].values
    return dist

def get_rssi_data(input):

    Datafile = input['Datafile']                    #input the datafile
    data = pd.read_csv(Datafile)                    #load the datafile and create a dataframe           
    length = len(data)
    columns = data.columns.values
    rssi_columns = columns[4:]
    rssi_data = data[rssi_columns]
    rssi_data = np.array(rssi_data)
    # shape = np.append(rssi_data.shape, 1)
    # rssi_data = rssi_data.reshape(shape)
    return rssi_data


def get_eta_data(input):

    Datafile = input['Datafile']                    #input the datafile
    data = pd.read_csv(Datafile)                    #load the datafile and create a dataframe           
    length = len(data)
    columns = data.columns.values

    train_dataframe = pd.DataFrame()

    x_router = [ROUTER_POS[0]]*len(data)
    y_router = [ROUTER_POS[1]]*len(data)

    train_data_columns = ['eta_1', 'eta_2', 'eta_3', 'eta_4']
    train_dataframe['rssi_0'] = data['rssi_0']
    
    train_dataframe['eta_1'] = compute_eta(data[columns[5]], 1)
    train_dataframe['eta_2'] = compute_eta(data[columns[6]], 2)
    train_dataframe['eta_3'] = compute_eta(data[columns[7]], 3)
    train_dataframe['eta_4'] = compute_eta(data[columns[8]], 4)
    # for i in range(4):
    #   train_dataframe[train_data_columns[i]] = compute_eta(data[columns[i+5]], i+1)

        
    # print 'normalizing data... '  
    # scaler = StandardScaler()  
    # scaler.fit(train_dataframe)  
    # train_dataframe = scaler.transform(train_dataframe) 
    # print 'Normalization done'

    # print 'logging normalized data'
    # normalized_data = pd.DataFrame(train_dataframe)
    # normalized_data['distance'] = Tvar_dataframe
    # normalized_data.to_csv('normalized_trainingData.csv')
    # print 'normalized data saved in normalized_trainingData.csv'

    return train_dataframe.values 


def data_cleaning(dataframe):
    # for i in range(len(dataframe)):

    #   row = dataframe.loc[i:i]
    print "data cleaning initiated"
    
    print "checking values of x and y coordinates"
    dataframe = dataframe[dataframe.x < 999]
    dataframe = dataframe[dataframe.y < 999]
    print "checked"

    dataframe = dataframe[dataframe.obj0 + dataframe.obj1 + dataframe.obj2 + dataframe.obj3 + dataframe.obj4 < 11]
    dataframe = dataframe[dataframe.obj0 + dataframe.obj1 + dataframe.obj2 + dataframe.obj3 + dataframe.obj4 > -1]
    
    print "checking ids of nodeMCU's"
    dataframe = dataframe[dataframe.obj0 < 6]
    dataframe = dataframe[dataframe.obj1 < 6]
    dataframe = dataframe[dataframe.obj2 < 6]
    dataframe = dataframe[dataframe.obj3 < 6]
    dataframe = dataframe[dataframe.obj4 < 6]

    dataframe = dataframe[dataframe.obj0 > -1]
    dataframe = dataframe[dataframe.obj1 > -1]
    dataframe = dataframe[dataframe.obj2 > -1]
    dataframe = dataframe[dataframe.obj3 > -1]
    dataframe = dataframe[dataframe.obj4 > -1]
    print 'checked'

    print "checking errors in received datapacket"
    dataframe = dataframe[dataframe.obj0 != dataframe.obj1]
    dataframe = dataframe[dataframe.obj0 != dataframe.obj2]
    dataframe = dataframe[dataframe.obj0 != dataframe.obj3]
    dataframe = dataframe[dataframe.obj0 != dataframe.obj4]
    
    dataframe = dataframe[dataframe.obj1 != dataframe.obj2]
    dataframe = dataframe[dataframe.obj1 != dataframe.obj3]
    dataframe = dataframe[dataframe.obj1 != dataframe.obj4]
    
    dataframe = dataframe[dataframe.obj2 != dataframe.obj3]
    dataframe = dataframe[dataframe.obj2 != dataframe.obj4]

    dataframe = dataframe[dataframe.obj3 != dataframe.obj4]
    print 'checked'


    print "dataset filtered"
    return dataframe

    
def train(input, sort_flag, avg_flag):
    Datafile = input['Datafile']                    #input the datafile
    data = pd.read_csv(Datafile)                    #load the datafile and create a dataframe           


    # Define data and labels 
    train_dataframe = pd.DataFrame()
    Tvar_dataframe = pd.DataFrame()
    if sort_flag:
        RSSI_columns = ['distance', 'dt', 'T', 'rssi_0', 'rssi_1', 'rssi_2', 'rssi_3', 'rssi_4']
        #train_dataframe  = pd.DataFrame(data[RSSI_columns[3:]])
        x_router = [router_pos[0]]*len(data)
        y_router = [router_pos[1]]*len(data)
        train_data_columns = ['eta_1', 'eta_2', 'eta_3', 'eta_4']
        train_dataframe['rssi_0'] = data['rssi_0']
        for i in range(4):
            train_dataframe[train_data_columns[i]] =  compute_eta(data[RSSI_columns[i + 4]],1+i)

# def data_plot(dataframe):
#   plt.plot(dataframe)



def getbatch_X(step, X):
    # X = X.values
    data = np.zeros((hparams.batch_size, hparams.num_time, hparams.num_nodes))
    
    for i in range(hparams.batch_size):
        for j in range(hparams.num_time):
            # import pdb; pdb.set_trace()
            data[i][j] = X[step*hparams.batch_size + i + j].reshape(5)

    return data     


def getbatch_labels(step, dist):
    data = np.zeros(hparams.batch_size)
    for i in range(hparams.batch_size):
        data[i] = dist[step*hparams.batch_size + i + hparams.num_time - 1]
    return data     


def getX_test(X):

    test_shape = np.shape(X)
    test_length = test_shape[0]
    num_inputs = int(test_length/hparams.num_time)

    data = np.zeros((num_inputs, hparams.num_time, hparams.num_nodes))
    for i in range(num_inputs):
        for j in range(hparams.num_time):
            data[i][j] = X[i*hparams.num_time + j]
    return data     

def gettest_labels(dist):
    test_length = (np.shape(dist))[0]
    num_inputs = int(test_length/hparams.num_time)

    data = np.zeros((num_inputs, hparams.num_time))
    for i in range(num_inputs):
        for j in range(hparams.num_time):
            data[i][j] = dist[i*hparams.num_time + j]
    return data     

def data_stats(dataframe):
        
    print 'normalizing data... '    
    scaler = StandardScaler()  
    scaler.fit(dataframe)  
    train_dataframe = scaler.transform(train_dataframe) 
    print 'Normalization done'

    print 'logging normalized data'
    normalized_data = pd.DataFrame(train_dataframe)
    normalized_data['distance'] = Tvar_dataframe
    normalized_data.to_csv('normalized_trainingData.csv')
    print 'normalized data saved in normalized_trainingData.csv'
    reg = MLPRegressor(solver='lbfgs')

# data_path = "../data/sorted_data/pos1_sorted.csv"
# inputs = {"Datafile":  data_path}           # Get the data filepath 
# dist = get_dist_data(inputs)                  # Get array containing distance of the object node to router 
# X = get_rssi_data(inputs)                     # Get RSSI data. Shape =  

