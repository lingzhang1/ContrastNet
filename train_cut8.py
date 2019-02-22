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
parser.add_argument('--num_point', type=int, default=128, help='Point Number [256/512/1024/2048] [default: 1024]')
parser.add_argument('--max_epoch', type=int, default=250, help='Epoch to run [default: 250]')
parser.add_argument('--batch_size', type=int, default=120, help='Batch Size during training [default: 32]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=200000, help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=0.7, help='Decay rate for lr decay [default: 0.8]')
parser.add_argument('--model_path', default='log/model.ckpt', help='model checkpoint file path [default: log/model.ckpt]')

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
MODEL_PATH = FLAGS.model_path

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
    os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048_cut8/train_files.txt'))
TEST_FILES = provider.getDataFiles(\
    os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048_cut8/test_files.txt'))

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

def get_random_points(points):
    total = []
    for i in range(len(points)):
        idx = np.random.randint(points[i].shape[0], size=NUM_POINT)
        total.append(points[i][idx,:])
    return total

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
            pred, feature1, feature2, end_points = MODEL.get_model(pointclouds_pl_1, pointclouds_pl_2, is_training_pl, bn_decay=bn_decay)
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

        # saver.restore(sess, tf.train.latest_checkpoint(MODEL_PATH))
        # saver.restore(sess, MODEL_PATH)
        # log_string("Model restored.")

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
            if epoch % 2 == 0:
                save_path = saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"))
                log_string("Model saved in file: %s" % save_path)

def train_one_epoch(sess, ops, train_writer):
    """ ops: dict mapping from string to tf ops """
    is_training = True

    # Shuffle train files
    train_file_idxs = np.arange(0, len(TRAIN_FILES))
    np.random.shuffle(train_file_idxs)

    current_data_1 = np.empty([60*len(TRAIN_FILES), NUM_POINT, 3], dtype=float)
    current_data_2 = np.empty([60*len(TRAIN_FILES), NUM_POINT, 3], dtype=float)
    current_label = np.empty([60*len(TRAIN_FILES),1], dtype=int)

    fn = 0
    count = 0
    while fn < len(TRAIN_FILES) - 1:
        # log_string('----' + str(fn) + '-----')
        print(TRAIN_FILES[train_file_idxs[fn]])
        total_a, _ = provider.loadDataFile_cut(TRAIN_FILES[train_file_idxs[fn]])
        total_a = get_random_points(total_a)

        fn = fn + 1

        total_b, _ = provider.loadDataFile_cut(TRAIN_FILES[train_file_idxs[fn]])
        total_b = get_random_points(total_b)

        fn = fn + 1

        label_1 = [[2, 5], [3, 5], [1, 6], [1, 7], [2, 8], [3, 8], [4, 7], [4, 6], [5, 8], [1, 4], [2, 3], [6, 7]]
        label_2 = [[4, 5], [1, 8], [2, 7], [3, 6]]
        label_3 = [[5, 7], [1, 3], [1, 5], [2, 4], [2, 6], [4, 8], [6, 8], [3, 7]]
        label_4 = [[5, 6], [1, 2], [3, 4], [7, 8]]

        pair_num = 0
        for index in range(len(label_1)):
                current_data_1[120*count+pair_num,:,:] = total_a[label_1[index][0] - 1]
                current_data_2[120*count+pair_num, :,:] = total_a[label_1[index][1] - 1]
                current_label[120*count+pair_num,:] = 1
                pair_num = pair_num + 1
                current_data_1[120*count+pair_num,:,:] = total_b[label_1[index][0] - 1]
                current_data_2[120*count+pair_num, :,:] = total_b[label_1[index][1] - 1]
                current_label[120*count+pair_num,:] = 1
                pair_num = pair_num + 1
        for index in range(len(label_2)):
                current_data_1[120*count+pair_num,:,:] = total_a[label_2[index][0] - 1]
                current_data_2[120*count+pair_num, :,:] = total_a[label_2[index][1] - 1]
                current_label[120*count+pair_num,:] = 2
                pair_num = pair_num + 1
                current_data_1[120*count+pair_num,:,:] = total_b[label_2[index][0] - 1]
                current_data_2[120*count+pair_num, :,:] = total_b[label_2[index][1] - 1]
                current_label[120*count+pair_num,:] = 2
                pair_num = pair_num + 1
        for index in range(len(label_3)):
                current_data_1[120*count+pair_num,:,:] = total_a[label_3[index][0] - 1]
                current_data_2[120*count+pair_num, :,:] = total_a[label_3[index][1] - 1]
                current_label[120*count+pair_num,:] = 3
                pair_num = pair_num + 1
                current_data_1[120*count+pair_num,:,:] = total_b[label_3[index][0] - 1]
                current_data_2[120*count+pair_num, :,:] = total_b[label_3[index][1] - 1]
                current_label[120*count+pair_num,:] = 3
                pair_num = pair_num + 1
        for index in range(len(label_4)):
                current_data_1[120*count+pair_num,:,:] = total_a[label_4[index][0] - 1]
                current_data_2[120*count+pair_num, :,:] = total_a[label_4[index][1] - 1]
                current_label[120*count+pair_num,:] = 4
                pair_num = pair_num + 1
                current_data_1[120*count+pair_num,:,:] = total_b[label_4[index][0] - 1]
                current_data_2[120*count+pair_num, :,:] = total_b[label_4[index][1] - 1]
                current_label[120*count+pair_num,:] = 4
                pair_num = pair_num + 1
        # label 0
        for index in range(len(total_a)):
            for index2 in range(len(total_b)):
                current_data_1[120*count+pair_num,:,:] = total_a[index]
                current_data_2[120*count+pair_num, :,:] = total_b[index2]
                current_label[120*count+pair_num,:] = 0
                pair_num = pair_num + 1
        count = count + 1

    print(count)
    current_label = np.squeeze(current_label)

    file_size = current_data_1.shape[0]
    num_batches = file_size // BATCH_SIZE

    total_correct = 0
    total_seen = 0
    loss_sum = 0

    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx+1) * BATCH_SIZE

        # shuffle each batch
        data_1 = current_data_1[start_idx:end_idx, :, :]
        data_2 = current_data_2[start_idx:end_idx, :, :]
        label = current_label[start_idx:end_idx]
        combine_data = np.concatenate((data_1, data_2), axis=2)
        combine_data, label, _ = provider.shuffle_data(combine_data, np.squeeze(label))
        data_1 = combine_data[:, :, 0:3]
        data_2 = combine_data[:, :, 3:6]
        label = np.squeeze(label)

        # Augment batched point clouds by rotation and jittering
        rotated_data_1 = provider.rotate_point_cloud(data_1)
        jittered_data_1 = provider.jitter_point_cloud(rotated_data_1)
        jittered_data_1 = provider.random_scale_point_cloud(jittered_data_1)
        jittered_data_1 = provider.rotate_perturbation_point_cloud(jittered_data_1)
        jittered_data_1 = provider.shift_point_cloud(jittered_data_1)

        rotated_data_2 = provider.rotate_point_cloud(data_2)
        jittered_data_2 = provider.jitter_point_cloud(rotated_data_2)
        jittered_data_2 = provider.random_scale_point_cloud(jittered_data_2)
        jittered_data_2 = provider.rotate_perturbation_point_cloud(jittered_data_2)
        jittered_data_2 = provider.shift_point_cloud(jittered_data_2)

        feed_dict = {ops['pointclouds_pl_1']: jittered_data_1,
                     ops['pointclouds_pl_2']: jittered_data_2,
                     ops['labels_pl']: label,
                     ops['is_training_pl']: is_training,}
        summary, step, _, loss_val, pred_val = sess.run([ops['merged'], ops['step'],
            ops['train_op'], ops['loss'], ops['pred']], feed_dict=feed_dict)
        train_writer.add_summary(summary, step)
        pred_val = np.argmax(pred_val, 1)
        correct = np.sum(pred_val == label)
        total_correct += correct
        total_seen += BATCH_SIZE
        loss_sum += loss_val
        if batch_idx % 50 == 0:
            # '{0:2d} {1:3d} {2:4d}'.format(x, x*x, x*x*x)
            log_string('mean loss: {0:f}     accuracy: {1:f}'.format(loss_sum / float(batch_idx+1), total_correct / float(total_seen)))
            # log_string('mean loss: %f' % (loss_sum / float(batch_idx+1)))
            # log_string('accuracy: %f' % (total_correct / float(total_seen)))

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
