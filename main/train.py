import datetime
import os
import sys
import time
import cv2

import tensorflow as tf
import numpy as np

sys.path.append(os.getcwd())
from tensorflow.contrib import slim
from nets import model_train as model
from utils.dataset import data_provider as data_provider
from utils.rpn_msr.proposal_layer import proposal_layer
from utils.text_connector.detectors import TextDetector

tf.app.flags.DEFINE_float('learning_rate', 1e-5, '')
tf.app.flags.DEFINE_integer('max_steps', 50000, '')
tf.app.flags.DEFINE_integer('decay_steps', 30000, '')
tf.app.flags.DEFINE_float('decay_rate', 0.1, '')
tf.app.flags.DEFINE_float('moving_average_decay', 0.997, '')
tf.app.flags.DEFINE_integer('num_readers', 4, '')
tf.app.flags.DEFINE_string('gpu', '0', '')
tf.app.flags.DEFINE_string('checkpoint_path_to_resume', '/floyd/input/checkpoints_mlt', '')
tf.app.flags.DEFINE_string('checkpoint_path', 'checkpoints_mlt/', '')
tf.app.flags.DEFINE_string('logs_path', 'logs_mlt/', '')
tf.app.flags.DEFINE_string('pretrained_model_path', 'data/vgg_16.ckpt', '')
tf.app.flags.DEFINE_boolean('restore', True, '')
tf.app.flags.DEFINE_integer('save_checkpoint_steps', 2000, '')
tf.app.flags.DEFINE_string('data_folder', '/floyd/input/checkpoints_mlt', '')
tf.app.flags.DEFINE_string('data_eval_folder', '/floyd/input/naapf', '')
FLAGS = tf.app.flags.FLAGS

def get_iou(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou


def plain_iou(bboxes1, bboxes2):
        x11, y11, x12, y12 = np.split(bboxes1, 4, axis=0)
        x21, y21, x22, y22 = np.split(bboxes2, 4, axis=0)
        
        # Sanity check
        assert x11 <= x12
        assert y11 <= y12
        assert x21 <= x22
        assert y21 <= y22
        
        xA = np.maximum(x11, np.transpose(x21))
        yA = np.maximum(y11, np.transpose(y21))
        xB = np.minimum(x12, np.transpose(x22))
        yB = np.minimum(y12, np.transpose(y22))
        interArea = np.maximum((xB - xA + 1), 0) * np.maximum((yB - yA + 1), 0)
        boxAArea = (x12 - x11 + 1) * (y12 - y11 + 1)
        boxBArea = (x22 - x21 + 1) * (y22 - y21 + 1)
        iou = interArea / (boxAArea + np.transpose(boxBArea) - interArea)

        # Sanity check
        assert iou >= 0.0
        assert iou <= 1.0
        return iou


def main(argv=None):
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
    now = datetime.datetime.now()
    StyleTime = now.strftime("%Y-%m-%d-%H-%M-%S")
    os.makedirs(FLAGS.logs_path + StyleTime)
    if not os.path.exists(FLAGS.checkpoint_path):
        os.makedirs(FLAGS.checkpoint_path)

    input_image = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_image')
    input_bbox = tf.placeholder(tf.float32, shape=[None, 5], name='input_bbox')
    input_im_info = tf.placeholder(tf.float32, shape=[None, 3], name='input_im_info')

    global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
    learning_rate = tf.Variable(FLAGS.learning_rate, trainable=False)
    tf.summary.scalar('learning_rate', learning_rate)
    opt = tf.train.AdamOptimizer(learning_rate)

    gpu_id = int(FLAGS.gpu)
    with tf.device('/gpu:%d' % gpu_id):
        with tf.name_scope('model_%d' % gpu_id) as scope:
            bbox_pred, cls_pred, cls_prob = model.model(input_image)
            total_loss, model_loss, rpn_cross_entropy, rpn_loss_box = model.loss(bbox_pred, cls_pred, input_bbox,
                                                                                 input_im_info)
            batch_norm_updates_op = tf.group(*tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope))
            grads = opt.compute_gradients(total_loss)

    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    summary_op = tf.summary.merge_all()
    variable_averages = tf.train.ExponentialMovingAverage(
        FLAGS.moving_average_decay, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())
    with tf.control_dependencies([variables_averages_op, apply_gradient_op, batch_norm_updates_op]):
        train_op = tf.no_op(name='train_op')

    saver = tf.train.Saver(tf.global_variables(), max_to_keep=100)
    summary_writer = tf.summary.FileWriter(FLAGS.logs_path + StyleTime, tf.get_default_graph())

    init = tf.global_variables_initializer()

    if FLAGS.pretrained_model_path is not None:
        variable_restore_op = slim.assign_from_checkpoint_fn(FLAGS.pretrained_model_path,
                                                             slim.get_trainable_variables(),
                                                             ignore_missing_vars=True)

    steps_in_epoch = data_provider.get_steps_in_epoch(FLAGS.data_folder)
    epoch = 0
    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.95
    config.allow_soft_placement = True
    with tf.Session(config=config) as sess:
        if FLAGS.restore:
            ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_path_to_resume)
            restore_step = int(ckpt.split('.')[0].split('_')[-1])
            print("continue training from previous checkpoint {}".format(restore_step))
            saver.restore(sess, ckpt)
        else:
            sess.run(init)
            restore_step = 0
            if FLAGS.pretrained_model_path is not None:
                variable_restore_op(sess)

        data_generator = data_provider.get_batch(FLAGS.data_folder, num_workers=FLAGS.num_readers)
        start = time.time()
        for step in range(restore_step, FLAGS.max_steps):
            data = next(data_generator)
            ml, tl, _, summary_str = sess.run([model_loss, total_loss, train_op, summary_op],
                                              feed_dict={input_image: data[0],
                                                         input_bbox: data[1],
                                                         input_im_info: data[2]})

            summary_writer.add_summary(summary_str, global_step=step)

            if step != 0 and step % FLAGS.decay_steps == 0:
                sess.run(tf.assign(learning_rate, learning_rate.eval() * FLAGS.decay_rate))

            if step % 10 == 0:
                avg_time_per_step = (time.time() - start) / 10
                start = time.time()
                print('Step {:06d}, model loss {:.4f}, total loss {:.4f}, {:.2f} seconds/step, LR: {:.6f}'.format(
                    step, ml, tl, avg_time_per_step, learning_rate.eval()))

            if (step + 1) % FLAGS.save_checkpoint_steps == 0:
                filename = ('ctpn_{:d}'.format(step + 1) + '.ckpt')
                filename = os.path.join(FLAGS.checkpoint_path, filename)
                saver.save(sess, filename)
                print('Write model to: {:s}'.format(filename))
                
            if step % steps_in_epoch == 0:
                epoch += 1

            
#             if step % steps_in_epoch == 0:
            if step % 2000 == 0:
                data_eval_generator = data_provider.get_batch(FLAGS.data_eval_folder, num_workers=FLAGS.num_readers)   
                iou_list = []
                
                for it in range(steps_in_epoch):
                    gt_bboxes = []
                    pred_bboxes = []
                    data = next(data_eval_generator)
                    
                    # Adding Ground Truth Label
                    img_to_log = data[0][0]
                    for p in data[1]:
                        cv2.rectangle(img_to_log, (p[0], p[1]), (p[2], p[3]), color=(0, 0, 255), thickness=1)
                    
                     # Debugging
                    img_to_log= np.reshape(img_to_log, (-1, 228, 228, 3))
                    image_op = tf.summary.image("Eval data", img_to_log)
                    
                    bbox_pred_val, cls_prob_val, image_str = sess.run([bbox_pred, cls_prob, image_op],
                                                       feed_dict={input_image: data[0],
                                                                  input_im_info: data[2]})
                    
                    summary_writer.add_summary(image_str, global_step=step)
                    
                    textsegs, _ = proposal_layer(cls_prob_val, bbox_pred_val, data[2])
                    scores = textsegs[:, 0]
                    textsegs = textsegs[:, 1:5]

                    textdetector = TextDetector(DETECT_MODE='H')
                    
                    img = data[0][0]
                    img_to_log_w_bboxes = data[0][0]
                    boxes = textdetector.detect(textsegs, scores[:, np.newaxis], img.shape[:2])
                    boxes = np.array(boxes, dtype=np.int)
                    for b in boxes:
                        pred_bboxes.append([b[0], b[1], b[6], b[7]])
                        cv2.rectangle(img_to_log_w_bboxes, (b[0], b[1]), (b[2], b[3]), color=(255, 0, 0), thickness=1)
                    
                    tuple_points = data[1]
                    for point in tuple_points:
                        gt_bboxes.append([point[0], point[1], point[2], point[3]])
                    
                     # Debugging
                    img_to_log2 = np.reshape(img_to_log_w_bboxes, (-1, 228, 228, 3))
                    image_op2 = tf.summary.image("Eval data w/ bbox", img_to_log2)
                    
                    image_str2 = sess.run(image_op2)
                    summary_writer.add_summary(image_str2, global_step=step)
                    

                    # Extend list to meet the one for gt
                    if len(pred_bboxes) == 0 or len(gt_bboxes) > len(pred_bboxes):
                        pred_bboxes = pred_bboxes[:len(gt_bboxes)] + [[0, 0, 0, 0]]*(len(gt_bboxes) - len(pred_bboxes))

  
                    for bb_gt, bb_pred in zip(gt_bboxes, pred_bboxes):
                        bb1 = np.array(bb_gt, dtype=np.int)
                        bb2 = np.array(bb_pred, dtype=np.int)
                        iou_list.append(plain_iou(np.array(bb1, dtype=np.int), np.array(bb2, dtype=np.int)))
                            
                       
                print('# Eval # Step {:06d}, Epoch {:03d}, mIoU score = {:.3f}'.format(step, epoch, np.average(iou_list)))
                        

if __name__ == '__main__':
    tf.app.run()
