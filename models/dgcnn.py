import tensorflow as tf
import numpy as np
import math
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
sys.path.append(os.path.join(BASE_DIR, '../../utils'))
import tf_util
from transform_nets import input_transform_net


def placeholder_inputs(batch_size, num_point):
  pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))
  labels_pl = tf.placeholder(tf.int32, shape=(batch_size))
  return pointclouds_pl, labels_pl

def model(point_cloud, is_training, cut, bn_decay=None):
  """ Classification PointNet, input is BxNx3, output Bx40 """
  k = 20

  adj_matrix = tf_util.pairwise_distance(point_cloud)
  nn_idx = tf_util.knn(adj_matrix, k=k)
  edge_feature = tf_util.get_edge_feature(point_cloud, nn_idx=nn_idx, k=k)

  # with tf.variable_scope('transform_net1') as sc:
    # transform = input_transform_net(edge_feature, is_training, cut, bn_decay, K=3)

  point_cloud_transformed = tf.matmul(point_cloud, edge_feature)
  adj_matrix = tf_util.pairwise_distance(point_cloud_transformed)
  nn_idx = tf_util.knn(adj_matrix, k=k)
  edge_feature = tf_util.get_edge_feature(point_cloud_transformed, nn_idx=nn_idx, k=k)

  net = tf_util.conv2d(edge_feature, 128, [1,1],
                       padding='VALID', stride=[1,1],
                       bn=True, is_training=is_training,
                       scope=cut+'dgcnn1', bn_decay=bn_decay)
  net = tf.reduce_max(net, axis=-2, keep_dims=True)
  net1 = net

  adj_matrix = tf_util.pairwise_distance(net)
  nn_idx = tf_util.knn(adj_matrix, k=k)
  edge_feature = tf_util.get_edge_feature(net, nn_idx=nn_idx, k=k)

  net = tf_util.conv2d(edge_feature, 128, [1,1],
                       padding='VALID', stride=[1,1],
                       bn=True, is_training=is_training,
                       scope=cut+'dgcnn2', bn_decay=bn_decay)
  net = tf.reduce_max(net, axis=-2, keep_dims=True)
  net2 = net

  adj_matrix = tf_util.pairwise_distance(net)
  nn_idx = tf_util.knn(adj_matrix, k=k)
  edge_feature = tf_util.get_edge_feature(net, nn_idx=nn_idx, k=k)

  net = tf_util.conv2d(edge_feature, 128, [1,1],
                       padding='VALID', stride=[1,1],
                       bn=True, is_training=is_training,
                       scope=cut+'dgcnn3', bn_decay=bn_decay)
  net = tf.reduce_max(net, axis=-2, keep_dims=True)
  net3 = net

  adj_matrix = tf_util.pairwise_distance(net)
  nn_idx = tf_util.knn(adj_matrix, k=k)
  edge_feature = tf_util.get_edge_feature(net, nn_idx=nn_idx, k=k)

  net = tf_util.conv2d(edge_feature, 256, [1,1],
                       padding='VALID', stride=[1,1],
                       bn=True, is_training=is_training,
                       scope=cut+'dgcnn4', bn_decay=bn_decay)
  net = tf.reduce_max(net, axis=-2, keep_dims=True)
  net4 = net

  net = tf_util.conv2d(tf.concat([net1, net2, net3, net4], axis=-1), 512, [1, 1],
                       padding='VALID', stride=[1,1],
                       bn=True, is_training=is_training,
                       scope=cut+'agg', bn_decay=bn_decay)

  net = tf.reduce_max(net, axis=1, keep_dims=True)
  return net


def get_model(point_cloud_1, point_cloud_2, is_training, bn_decay=None):

  batch_size = point_cloud_1.get_shape()[0].value
  num_point = point_cloud_1.get_shape()[1].value
  end_points = {}

  net1 = model(point_cloud_1, is_training, '1', bn_decay=None)
  net2 = model(point_cloud_2, is_training, '2', bn_decay=None)

  net = tf.concat([net1, net2], 3)

  net1 = tf.reshape(net1, [batch_size, -1])
  net2 = tf.reshape(net2, [batch_size, -1])

  # MLP on global point cloud vector
  net = tf.reshape(net, [batch_size, -1])
  net = tf_util.fully_connected(net, 1024, bn=True, is_training=is_training,
                                scope='fc1', bn_decay=bn_decay)
  net = tf_util.dropout(net, keep_prob=0.5, is_training=is_training,
                         scope='dp1')
  net = tf_util.fully_connected(net, 512, bn=True, is_training=is_training,
                                scope='fc2', bn_decay=bn_decay)
  net = tf_util.dropout(net, keep_prob=0.5, is_training=is_training,
                        scope='dp2')
  net = tf_util.fully_connected(net, 2, activation_fn=None, scope='fc3')

  return net, net1, net2, end_points
  # return net1, net2, end_points


def get_loss(pred, label, end_points):
  """ pred: B*NUM_CLASSES,
      label: B, """
  labels = tf.one_hot(indices=label, depth=2)
  loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=pred, label_smoothing=0.2)
  classify_loss = tf.reduce_mean(loss)
  return classify_loss

# contrastive loss
def get_constra_loss(feat1, feat2, label, end_points):
  """ feat1: B*256,
      feat2: B*256,
      label: B, """
  feat1 = tf.math.l2_normalize(feat1, axis=0, epsilon=1e-12, name=None, dim=None)
  feat2 = tf.math.l2_normalize(feat2, axis=0, epsilon=1e-12, name=None, dim=None)
  contrastive_loss = tf.contrib.losses.metric_learning.contrastive_loss(label, feat1, feat2, margin=1.0)
  return contrastive_loss

def get_constra_cross_loss(pred, feat1, feat2, label, end_points):
    constrastive_loss = get_constra_loss(feat1, feat2, label, end_points)
    cross_entropy_loss = get_loss(pred, label, end_points)
    return constrastive_loss + 2 * cross_entropy_loss

if __name__=='__main__':
  batch_size = 2
  num_pt = 124
  pos_dim = 3

  input_feed = np.random.rand(batch_size, num_pt, pos_dim)
  label_feed = np.random.rand(batch_size)
  label_feed[label_feed>=0.5] = 1
  label_feed[label_feed<0.5] = 0
  label_feed = label_feed.astype(np.int32)

  # # np.save('./debug/input_feed.npy', input_feed)
  # input_feed = np.load('./debug/input_feed.npy')
  # print input_feed

  with tf.Graph().as_default():
    input_pl, label_pl = placeholder_inputs(batch_size, num_pt)
    pos, ftr = get_model(input_pl, tf.constant(True))
    # loss = get_loss(logits, label_pl, None)

    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      feed_dict = {input_pl: input_feed, label_pl: label_feed}
      res1, res2 = sess.run([pos, ftr], feed_dict=feed_dict)
      print res1.shape
      print res1

      print res2.shape
      print res2
