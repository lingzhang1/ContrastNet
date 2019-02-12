import argparse
import math
import h5py
import numpy as np
import tensorflow as tf
import socket
import importlib
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
import provider
import tf_util

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='dgcnn', help='Model name: dgcnn')
parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
parser.add_argument('--num_point', type=int, default=512, help='Point Number [256/512/1024/2048] [default: 1024]')
parser.add_argument('--max_epoch', type=int, default=250, help='Epoch to run [default: 250]')
parser.add_argument('--batch_size', type=int, default=32, help='Batch Size during training [default: 32]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=200000, help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=0.7, help='Decay rate for lr decay [default: 0.8]')
FLAGS = parser.parse_args()


BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
GPU_INDEX = FLAGS.gpu
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate

MODEL = importlib.import_module(FLAGS.model) # import network module
MODEL_FILE = os.path.join(BASE_DIR, 'models', FLAGS.model+'.py')
LOG_DIR = FLAGS.log_dir
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
os.system('cp %s %s' % (MODEL_FILE, LOG_DIR)) # bkp of model def
os.system('cp train.py %s' % (LOG_DIR)) # bkp of train procedure
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

MAX_NUM_POINT = 2048
NUM_CLASSES = 40

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

HOSTNAME = socket.gethostname()

# ModelNet40 official train/test split
TRAIN_FILES = provider.getDataFiles( \
    os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048_cut/train_files.txt'))
TEST_FILES = provider.getDataFiles(\
    os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048_cut/test_files.txt'))

def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)


def get_learning_rate(batch):
    learning_rate = tf.train.exponential_decay(
                        BASE_LEARNING_RATE,  # Base learning rate.
                        batch * BATCH_SIZE,  # Current index into the dataset.
                        DECAY_STEP,          # Decay step.
                        DECAY_RATE,          # Decay rate.
                        staircase=True)
    learning_rate = tf.maximum(learning_rate, 0.00001) # CLIP THE LEARNING RATE!
    return learning_rate

def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
                      BN_INIT_DECAY,
                      batch*BATCH_SIZE,
                      BN_DECAY_DECAY_STEP,
                      BN_DECAY_DECAY_RATE,
                      staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay

def train():
    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(GPU_INDEX)):
            pointclouds_pl_1, labels_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT)
            pointclouds_pl_2, labels_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT)
            is_training_pl = tf.placeholder(tf.bool, shape=())
            print(is_training_pl)

            # Note the global_step=batch parameter to minimize.
            # That tells the optimizer to helpfully increment the 'batch' parameter for you every time it trains.
            batch = tf.Variable(0)
            bn_decay = get_bn_decay(batch)
            tf.summary.scalar('bn_decay', bn_decay)

            # Get model and loss
            pred, end_points = MODEL.get_model(pointclouds_pl_1, pointclouds_pl_2, is_training_pl, bn_decay=bn_decay)
            loss = MODEL.get_loss(pred, labels_pl, end_points)
            tf.summary.scalar('loss', loss)

            correct = tf.equal(tf.argmax(pred, 1), tf.to_int64(labels_pl))
            accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / float(BATCH_SIZE)
            tf.summary.scalar('accuracy', accuracy)

            # Get training operator
            learning_rate = get_learning_rate(batch)
            tf.summary.scalar('learning_rate', learning_rate)
            if OPTIMIZER == 'momentum':
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
            elif OPTIMIZER == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate)
            train_op = optimizer.minimize(loss, global_step=batch)

            # Add ops to save and restore all the variables.
            saver = tf.train.Saver()

        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)

        # Add summary writers
        #merged = tf.merge_all_summaries()
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'),
                                  sess.graph)
        test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test'))

        # Init variables
        init = tf.global_variables_initializer()
        # To fix the bug introduced in TF 0.12.1 as in
        # http://stackoverflow.com/questions/41543774/invalidargumenterror-for-tensor-bool-tensorflow-0-12-1
        #sess.run(init)
        sess.run(init, {is_training_pl: True})

        ops = {'pointclouds_pl_1': pointclouds_pl_1,
               'pointclouds_pl_2': pointclouds_pl_2,
               'labels_pl': labels_pl,
               'is_training_pl': is_training_pl,
               'pred': pred,
               'loss': loss,
               'train_op': train_op,
               'merged': merged,
               'step': batch}

        for epoch in range(MAX_EPOCH):
            log_string('**** EPOCH %03d ****' % (epoch))
            sys.stdout.flush()

            train_one_epoch(sess, ops, train_writer)
            # eval_one_epoch(sess, ops, test_writer)

            # Save the variables to disk.
            if epoch % 10 == 0:
                save_path = saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"))
                log_string("Model saved in file: %s" % save_path)



def train_one_epoch(sess, ops, train_writer):
    """ ops: dict mapping from string to tf ops """
    is_training = True

    # Shuffle train files
    train_file_idxs = np.arange(0, len(TRAIN_FILES))
    np.random.shuffle(train_file_idxs)

    current_data_1 = np.empty([len(TRAIN_FILES), 3, NUM_POINT])
    current_data_2 = np.empty([len(TRAIN_FILES), 3, NUM_POINT])
    current_label  =  np.empty([len(TRAIN_FILES),1])

    fn = 0
    count = 0
    while fn < len(TRAIN_FILES) - 1:
        # log_string('----' + str(fn) + '-----')
        a1, a2, a_label = provider.loadDataFile_cut(TRAIN_FILES[train_file_idxs[fn]])
        a1 = np.transpose(a1,(1,0))
        a2 = np.transpose(a2,(1,0))
        
        print('-----------------------------------')
        print(a1.shape)
        print(a2.shape)
        print(a_label.shape)
        print('-----------------------------------')
        if(a1.shape[1] < NUM_POINT):
            print('aaaaaaaaaaaaa')
            a1 = np.concatenate((a1, a1[0 : (NUM_POINT - a1.shape[1]), :]), axis=1)
        if(a2.shape[1] < NUM_POINT):
            print('bbbbbbbbbbbbbbbbbbb')
            a2 = np.concatenate((a2, a2[0 : (NUM_POINT - a2.shape[1]), :]), axis=1)

        a1 = a1[0:NUM_POINT,:]
        a2 = a2[0:NUM_POINT,:]
        # a_label = a_label[0:NUM_POINT]

        fn = fn + 1;

        b1, b2, b_label = provider.loadDataFile_cut(TRAIN_FILES[train_file_idxs[fn]])
        if(b1.shape[1] < NUM_POINT):
            b1 = np.concatenate((b1, b1[0 : (NUM_POINT - b1.shape[1]), :]), axis=0)
        if(b2.shape[1] < NUM_POINT):
            b2 = np.concatenate((b2, b2[0 : (NUM_POINT - b2.shape[1]), :]), axis=0)
        b1 = b1[0:NUM_POINT,:]
        b2 = b2[0:NUM_POINT,:]


        print('AAAAAAAAAAAAAAAAAAAAAAAAA')
        print(a1.shape)
        print(a2.shape)
        print(b1.shape)
        print(b2.shape)
        print(a_label.shape)
        print(b_label.shape)
        print('BBBBBBBBBBBBBBBBBBBBBBBBBBBB')



        fn = fn + 1;

        current_data_1[6*count,:,:] = a1
        current_data_2[6*count, :,:] = a2
        current_label[6*count,:] = 1

        current_data_1[6*count+1, :,:] = b1
        current_data_2[6*count+1, :,:] = b2
        current_label[6*count+1,:] = 1

        current_data_1[6*count+2, :,:] = a1
        current_data_2[6*count+2, :,:] = b1
        current_label[6*count+2,:] = 0

        current_data_1[6*count+3, :,:] = a1
        current_data_2[6*count+3, :,:] = b2
        current_label[6*count+3,:] = 0

        current_data_1[6*count+4, :,:] = a2
        current_data_2[6*count+4, :,:] = b1
        current_label[6*count+4,:] = 0

        current_data_1[6*count+5, :,:] = a2
        current_data_2[6*count+5, :,:] = b2
        current_label[6*count+5,:] = 0

        count = count + 1

    # current_data, current_label, _ = provider.shuffle_data(current_data, np.squeeze(current_label))
    # current_label = np.squeeze(current_label)

    file_size = current_data_1.shape[0]
    num_batches = file_size // BATCH_SIZE

    total_correct = 0
    total_seen = 0
    loss_sum = 0

    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx+1) * BATCH_SIZE

        # Augment batched point clouds by rotation and jittering
        rotated_data_1 = provider.rotate_point_cloud(current_data_1[start_idx:end_idx, :, :])
        jittered_data_1 = provider.jitter_point_cloud(rotated_data)
        jittered_data_1 = provider.random_scale_point_cloud(jittered_data)
        jittered_data_1 = provider.rotate_perturbation_point_cloud(jittered_data)
        jittered_data_1 = provider.shift_point_cloud(jittered_data)

        rotated_data_2 = provider.rotate_point_cloud(current_data_2[start_idx:end_idx, :, :])
        jittered_data_2 = provider.jitter_point_cloud(rotated_data)
        jittered_data_2 = provider.random_scale_point_cloud(jittered_data)
        jittered_data_2 = provider.rotate_perturbation_point_cloud(jittered_data)
        jittered_data_2 = provider.shift_point_cloud(jittered_data)

        feed_dict = {ops['pointclouds_pl_1']: jittered_data_1,
                     ops['pointclouds_pl_2']: jittered_data_2,
                     ops['labels_pl']: current_label[start_idx:end_idx],
                     ops['is_training_pl']: is_training,}
        summary, step, _, loss_val, pred_val = sess.run([ops['merged'], ops['step'],
            ops['train_op'], ops['loss'], ops['pred']], feed_dict=feed_dict)
        train_writer.add_summary(summary, step)
        pred_val = np.argmax(pred_val, 1)
        correct = np.sum(pred_val == current_label[start_idx:end_idx])
        total_correct += correct
        total_seen += BATCH_SIZE
        loss_sum += loss_val

        log_string('mean loss: %f' % (loss_sum / float(num_batches)))
        log_string('accuracy: %f' % (total_correct / float(total_seen)))


def eval_one_epoch(sess, ops, test_writer):
    """ ops: dict mapping from string to tf ops """
    is_training = False
    total_correct = 0
    total_seen = 0
    loss_sum = 0
    total_seen_class = [0 for _ in range(NUM_CLASSES)]
    total_correct_class = [0 for _ in range(NUM_CLASSES)]

    for fn in range(len(TEST_FILES)):
        log_string('----' + str(fn) + '-----')
        current_data, current_label = provider.loadDataFile(TEST_FILES[fn])
        current_data = current_data[:,0:NUM_POINT,:]
        current_label = np.squeeze(current_label)

        file_size = current_data.shape[0]
        num_batches = file_size // BATCH_SIZE

        for batch_idx in range(num_batches):
            start_idx = batch_idx * BATCH_SIZE
            end_idx = (batch_idx+1) * BATCH_SIZE

            feed_dict = {ops['pointclouds_pl']: current_data[start_idx:end_idx, :, :],
                         ops['labels_pl']: current_label[start_idx:end_idx],
                         ops['is_training_pl']: is_training}
            summary, step, loss_val, pred_val = sess.run([ops['merged'], ops['step'],
                ops['loss'], ops['pred']], feed_dict=feed_dict)
            pred_val = np.argmax(pred_val, 1)
            correct = np.sum(pred_val == current_label[start_idx:end_idx])
            total_correct += correct
            total_seen += BATCH_SIZE
            loss_sum += (loss_val*BATCH_SIZE)
            for i in range(start_idx, end_idx):
                l = current_label[i]
                total_seen_class[l] += 1
                total_correct_class[l] += (pred_val[i-start_idx] == l)

    log_string('eval mean loss: %f' % (loss_sum / float(total_seen)))
    log_string('eval accuracy: %f'% (total_correct / float(total_seen)))
    log_string('eval avg class acc: %f' % (np.mean(np.array(total_correct_class)/np.array(total_seen_class,dtype=np.float))))



if __name__ == "__main__":
    train()
    LOG_FOUT.close()
