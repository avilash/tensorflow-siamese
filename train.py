import os
import sys, getopt
import random
import numpy as np
import time
import tensorflow as tf 
import math
import cv2
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from load_data import generate_inhouse_datapairs, generate_apc_datapairs

# Defaults

VGG_NET = 1
RESNET50 = 2

APC = 1
INHOUSE = 2

IMAGENET_MEAN = [123.68, 116.78, 103.94]

batch_size = 16
num_workers = 4
num_epochs = 50
snapshot_epoch_step = 20
margin = 10.0
thresh = margin/2
weight_decay = 5e-4
ws = True 
net = RESNET50
last_layer_size = 1000
dataset = INHOUSE

should_evaluate_along_the_way = True


def _load_image(filename):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string, channels=3)          # (1)
    image = tf.cast(image_decoded, tf.float32)

    # smallest_side = 256.0
    # height, width = tf.shape(image)[0], tf.shape(image)[1]
    # height = tf.to_float(height)
    # width = tf.to_float(width)

    # scale = tf.cond(tf.greater(height, width),
    #                 lambda: smallest_side / width,
    #                 lambda: smallest_side / height)
    # new_height = tf.to_int32(height * scale)
    # new_width = tf.to_int32(width * scale)

    # resized_image = tf.image.resize_images(image, [new_height, new_width])  # (2)
    # crop_image = tf.random_crop(image, [224, 224, 3])                       # (3)

    side_length = 224
    resized_image = tf.image.resize_images(image, [side_length, side_length])
    means = tf.reshape(tf.constant(IMAGENET_MEAN), [1, 1, 3])
    centered_image = resized_image - means 
    if net == RESNET50:
        centered_image = centered_image/255.0                                   # (5)
    return centered_image

def _parse_function(filename, label):
    img1 = _load_image(filename[0])
    img2 = _load_image(filename[1])
    return tf.stack([img1 , img2]), label

def contrastive_loss(y,d):
    tmp = (1.0-y) * tf.square(d)
    tmp2 = y * tf.square(tf.maximum((margin - d),0))
    return tf.reduce_sum(tmp + tmp2)/batch_size/2.0

def contrastive_loss_np(y,d):
    tmp = (1.0-y) * np.square(d)
    tmp2 = y * np.square(np.maximum((margin - d),0))
    return np.sum(tmp + tmp2)/batch_size/2.0

def compute_accuracy(prediction, labels):
    accuracy = 0.0
    prediction_thresh = prediction.copy()
    prediction_thresh[prediction_thresh < thresh] = 0.0
    prediction_thresh[prediction_thresh >= thresh] = 1.0
    for i in range(prediction_thresh.shape[0]):
        if prediction_thresh[i] == labels[i]:
            accuracy += 1
    return accuracy/prediction_thresh.shape[0]

def train():
    # Set Batch Size
    if net == VGG_NET:
        batch_size = 16
    if net == RESNET50:
        batch_size = 128

    # Create Pairs for Siamese style Dataset
    if dataset == APC:
        tr_pairs, tr_y = generate_apc_datapairs("data" , "train-product-imgs.txt", "train-product-labels.txt", 41)
        te_pairs, te_y = generate_apc_datapairs("data" , "test-product-imgs.txt", "test-product-labels.txt", 61)
    if dataset == INHOUSE:
        tr_pairs, tr_y = generate_inhouse_datapairs("/home/dexf17/Work/HDD/Data/RCNN/30OBJ/Siamese" , "train")
        te_pairs, te_y = generate_inhouse_datapairs("/home/dexf17/Work/HDD/Data/RCNN/30OBJ/Siamese" , "test")
    print tr_pairs.shape
    print te_pairs.shape

    # Get batched dataset
    train_dataset = tf.contrib.data.Dataset.from_tensor_slices((tf.constant(tr_pairs) , tf.constant(tr_y)))
    train_dataset = train_dataset.shuffle(buffer_size=10000)  
    train_dataset = train_dataset.map(_parse_function, num_threads=num_workers, output_buffer_size=batch_size)
    batched_train_dataset = train_dataset.batch(batch_size)
    val_dataset = tf.contrib.data.Dataset.from_tensor_slices((tf.constant(te_pairs) , tf.constant(te_y)))
    val_dataset = val_dataset.shuffle(buffer_size=10000)  
    val_dataset = val_dataset.map(_parse_function, num_threads=num_workers, output_buffer_size=batch_size)
    batched_val_dataset = val_dataset.batch(batch_size)

    #Build Iterator
    iterator = tf.contrib.data.Iterator.from_structure(batched_train_dataset.output_types, batched_train_dataset.output_shapes)
    images, labels = iterator.get_next()
    train_iterator_init_op = iterator.make_initializer(batched_train_dataset)
    val_iterator_init_op = iterator.make_initializer(batched_val_dataset)

    # Build Model
    dropout_f = tf.placeholder("float")
    vgg = tf.contrib.slim.nets.vgg
    resnet_v1 = tf.contrib.slim.nets.resnet_v1
    image1 = images[:,0]
    image2 = images[:,1]

    if ws == True:
        with tf.variable_scope("net1") as scope:
            if net == VGG_NET:
                with slim.arg_scope(vgg.vgg_arg_scope(weight_decay=weight_decay)):
                    model1 , _ = vgg.vgg_16(image1, dropout_keep_prob=0.5, num_classes=last_layer_size)
            if net == RESNET50:
                with slim.arg_scope(resnet_v1.resnet_arg_scope()):
                    model1 , _ = resnet_v1.resnet_v1_50(image1, num_classes=last_layer_size)   
            scope.reuse_variables()
            if net == VGG_NET:
                with slim.arg_scope(vgg.vgg_arg_scope(weight_decay=weight_decay)):
                    model2 , _ = vgg.vgg_16(image2, dropout_keep_prob=0.5, num_classes=last_layer_size)
            if net == RESNET50:
                with slim.arg_scope(resnet_v1.resnet_arg_scope()):
                    model2 , _ = resnet_v1.resnet_v1_50(image2, num_classes=last_layer_size) 
    else:
        with tf.variable_scope("net1") as scope:
            if net == VGG_NET:
                with slim.arg_scope(vgg.vgg_arg_scope(weight_decay=weight_decay)):
                    model1 , _ = vgg.vgg_16(image1, dropout_keep_prob=0.5, num_classes=last_layer_size)
            if net == RESNET50:
                with slim.arg_scope(resnet_v1.resnet_arg_scope()):
                    model1 , _ = resnet_v1.resnet_v1_50(image1, num_classes=last_layer_size)    
        with tf.variable_scope("net2") as scope:
            if net == VGG_NET:
                with slim.arg_scope(vgg.vgg_arg_scope(weight_decay=weight_decay)):
                    model2 , _ = vgg.vgg_16(image2, dropout_keep_prob=0.5, num_classes=last_layer_size)
            if net == RESNET50:
                with slim.arg_scope(resnet_v1.resnet_arg_scope()):
                    model2 , _ = resnet_v1.resnet_v1_50(image2, num_classes=last_layer_size) 
    model1 = tf.reshape(model1 , [-1 , last_layer_size])
    model2 = tf.reshape(model2 , [-1 , last_layer_size])

    # Restore Weights
    if net == VGG_NET:
        variables_to_restore = slim.get_variables_to_restore(exclude=["net1/vgg_16/fc8","net2/vgg_16/fc8","net1/vgg_16/fc7","net2/vgg_16/fc7","net1/vgg_16/fc6","net2/vgg_16/fc6"])
        restorer = tf.train.Saver(variables_to_restore)
        fc8_1_variables = tf.contrib.framework.get_variables('net1/vgg_16/fc8')
        fc8_2_variables = tf.contrib.framework.get_variables('net2/vgg_16/fc8')
        fc7_1_variables = tf.contrib.framework.get_variables('net1/vgg_16/fc7')
        fc7_2_variables = tf.contrib.framework.get_variables('net2/vgg_16/fc7')
        fc6_1_variables = tf.contrib.framework.get_variables('net1/vgg_16/fc6')
        fc6_2_variables = tf.contrib.framework.get_variables('net2/vgg_16/fc6')
        fc8_1_init_op = tf.variables_initializer(fc8_1_variables)
        fc8_2_init_op = tf.variables_initializer(fc8_2_variables)
        fc7_1_init_op = tf.variables_initializer(fc7_1_variables)
        fc7_2_init_op = tf.variables_initializer(fc7_2_variables)
        fc6_1_init_op = tf.variables_initializer(fc6_1_variables)
        fc6_2_init_op = tf.variables_initializer(fc6_2_variables)
        # restorer = tf.train.Saver()
    if net == RESNET50:
        variables_to_restore = slim.get_variables_to_restore(exclude=["net1/resnet_v1_50/logits","net2/resnet_v1_50/logits"])
        restorer = tf.train.Saver(variables_to_restore)
        logits_1_variables = tf.contrib.framework.get_variables('net1/resnet_v1_50/logits')
        logits_2_variables = tf.contrib.framework.get_variables('net2/resnet_v1_50/logits')
        block4_1_variables = tf.contrib.framework.get_variables('net1/resnet_v1_50/block4')
        block4_2_variables = tf.contrib.framework.get_variables('net2/resnet_v1_50/block4')
        logits_1_init_op = tf.variables_initializer(logits_1_variables)
        logits_2_init_op = tf.variables_initializer(logits_2_variables)
        block4_1_init_op = tf.variables_initializer(block4_1_variables)
        block4_2_init_op = tf.variables_initializer(block4_2_variables)
        # restorer = tf.train.Saver()

    # Preparing variables to train
    variables_to_train = []
    variables_to_train_initializer = []
    if net == VGG_NET:
        variables_to_train = [fc8_1_variables , fc8_2_variables, fc7_1_variables , fc7_2_variables, fc6_1_variables , fc6_2_variables]
        variables_to_train_initializer = [fc8_1_init_op , fc8_2_init_op , fc7_1_init_op , fc7_2_init_op , fc6_1_init_op , fc6_2_init_op]
    if net == RESNET50:
        variables_to_train = [logits_1_variables , logits_2_variables , block4_1_variables , block4_2_variables]
        variables_to_train_initializer = [logits_1_init_op , logits_2_init_op]

    # Loss
    distance  = tf.sqrt(tf.reduce_sum(tf.pow(tf.subtract(model1, model2), 2), 1, keep_dims=True))
    distance = tf.cast(distance, tf.float32)
    # distance = tf.nn.l2_normalize(distance, 0, epsilon=1e-12, name=None)
    # distance = tf.div(tf.subtract(distance, tf.reduce_min(distance)), tf.subtract(tf.reduce_max(distance), tf.reduce_min(distance)))
    # distance =  distance * margin
    labels = tf.cast(labels, tf.float32)
    labels = tf.reshape(labels , [-1 , 1])
    loss = contrastive_loss(labels, distance)

    # Gradient Descent
    partial_optimizer = tf.train.AdamOptimizer(learning_rate = 0.0001)
    partial_train_op = partial_optimizer.minimize(loss, var_list=variables_to_train)
    # partial_optimizer = tf.train.GradientDescentOptimizer(1e-5)
    # partial_train_op = partial_optimizer.minimize(loss, var_list=variables_to_train)
    # full_optimizer = tf.train.AdamOptimizer(learning_rate = 0.0001)
    # full_train_op = full_optimizer.minimize(loss)
    # full_optimizer = tf.train.GradientDescentOptimizer(1e-5)
    # full_train_op = full_optimizer.minimize(loss)

    saver = tf.train.Saver()

    # Debug
    # for op in tf.get_default_graph().get_operations():
    #     print str(op.name) 
    
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.8
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        for init_op in variables_to_train_initializer:
            sess.run(init_op)

        if net == VGG_NET:
            if ws == True:
                restorer.restore(sess, "pretrained/vgg/siamese_ws_vgg_16.ckpt")
            else:
                restorer.restore(sess, "pretrained/vgg/siamese_vgg_16.ckpt")
        if net == RESNET50:
            if ws == True:
                restorer.restore(sess, "pretrained/resnet/siamese_ws_resnet_v1_50.ckpt")
            else:
                restorer.restore(sess, "pretrained/resnet/siamese_resnet_v1_50.ckpt")
        
        if (should_evaluate_along_the_way == False):
            print('-------------------------------------Training')
        for epoch in range(num_epochs):
            if (should_evaluate_along_the_way):
                if epoch % 5 == 0:
                    print('-------------------------------------Evaluating on Train Set')
                    sess.run(train_iterator_init_op)
                    predict = np.empty([0 , 1])
                    gt = np.empty([0 , 1])
                    while True:
                        try:
                            gt_tmp , predict_tmp = sess.run([labels , distance])
                            predict = np.append(predict , predict_tmp)
                            gt = np.append(gt , gt_tmp)
                        except tf.errors.OutOfRangeError:
                            break
                    tr_acc = compute_accuracy(predict, gt)
                    tr_loss = contrastive_loss_np(gt , predict)
                    predict = predict.flatten()
                    print('**Mean %0.5f' % (np.mean(predict)))
                    print('**Min - %0.5f Max -  %0.5f' % (np.min(predict) , np.max(predict)))
                    print('Train loss %0.5f' % (tr_loss/predict.shape[0]))  
                    print('Accuracy train set %0.5f' % (tr_acc*100))  
                    print('-------------------------------------Evaluating on Test Set')
                    sess.run(val_iterator_init_op)
                    predict = np.empty([0 , 1])
                    gt = np.empty([0 , 1])
                    while True:
                        try:
                            gt_tmp , predict_tmp = sess.run([labels , distance])
                            predict = np.append(predict , predict_tmp)
                            gt = np.append(gt , gt_tmp)
                        except tf.errors.OutOfRangeError:
                            break
                    te_acc = compute_accuracy(predict, gt)
                    te_loss = contrastive_loss_np(gt , predict)
                    predict = predict.flatten()
                    print('**Mean %0.5f' % (np.mean(predict)))
                    print('**Min - %0.5f Max -  %0.5f' % (np.min(predict) , np.max(predict)))
                    print('Test loss %0.5f' % (te_loss/predict.shape[0]))   
                    print('Accuracy test set %0.5f' % (te_acc*100))   
            if epoch % snapshot_epoch_step == 0:
                print('-------------------------------------Saving Checkpoint')
                # saver.save(sess, 'output/model',global_step=epoch)
            if (should_evaluate_along_the_way):
                if epoch % 5 == 0:
                    print('-------------------------------------Training')
            sess.run(train_iterator_init_op)
            avg_loss = 0.
            avg_acc = 0.
            avg_mean = 0.
            total_batch = 0
            while True:
                try:
                    _, loss_v, gt, predict = sess.run([partial_train_op, loss , labels , distance] , feed_dict={dropout_f:0.9})
                    avg_loss += loss_v
                    tr_acc = compute_accuracy(predict, gt)
                    avg_acc += tr_acc
                    avg_mean += np.mean(predict.flatten())
                    total_batch += 1
                except tf.errors.OutOfRangeError:
                    break
            print('**********epoch %d loss %0.5f acc %0.5f mean %0.5f' %(epoch+1, avg_loss/(total_batch), avg_acc/total_batch*100, avg_mean/total_batch))




        print('-------------------------------------Evaluating on Train Set')
        sess.run(train_iterator_init_op)
        predict = np.empty([0 , 1])
        gt = np.empty([0 , 1])
        while True:
            try:
                gt_tmp , predict_tmp = sess.run([labels , distance])
                predict = np.append(predict , predict_tmp)
                gt = np.append(gt , gt_tmp)
            except tf.errors.OutOfRangeError:
                break
        tr_acc = compute_accuracy(predict, gt)
        tr_loss = contrastive_loss_np(gt , predict)
        print('Train loss %0.5f' % (tr_loss/predict.shape[0]))  
        print('Accuracy train set %0.5f' % (tr_acc*100))  
        print('-------------------------------------Evaluating on Test Set')
        sess.run(val_iterator_init_op)
        predict = np.empty([0 , 1])
        gt = np.empty([0 , 1])
        while True:
            try:
                gt_tmp , predict_tmp = sess.run([labels , distance])
                predict = np.append(predict , predict_tmp)
                gt = np.append(gt , gt_tmp)
            except tf.errors.OutOfRangeError:
                break
        te_acc = compute_accuracy(predict, gt)
        te_loss = contrastive_loss_np(gt , predict)
        print('Test loss %0.5f' % (te_loss/predict.shape[0]))   
        print('Accuracy test set %0.5f' % (te_acc*100))   
        saver.save(sess, 'output/model',global_step=num_epochs)
        
def test(image1 , image2, model):

    # Set Batch Size
    batch_size = 10

    # Create Pairs for Siamese style Dataset
    te_pairs = [[image1, image2]]
    te_y = [1.]
    # te_pairs, te_y = generate_inhouse_datapairs("/home/dexf17/Work/HDD/Data/RCNN/30OBJ/Siamese" , "test")
    # te_pairs = []
    # te_y = []
    # for i in range(1,10):
    #     for j in range(1,10):
    #         te_pairs += [[str(i) + ".jpg", str(j) + ".jpg"]]
    #         if (i==j) or (i==6 and j==5) or (i==1 and j==2) or (i==2 and j==1) or (i==3 and j==4) or (i==4 and j==3) or (i==5 and j==6):
    #             te_y += [0.]
    #         else:
    #             te_y += [1.]

    # Get batched dataset
    val_dataset = tf.contrib.data.Dataset.from_tensor_slices((tf.constant(te_pairs) , tf.constant(te_y)))
    val_dataset = val_dataset.shuffle(buffer_size=10000)  
    val_dataset = val_dataset.map(_parse_function, num_threads=num_workers, output_buffer_size=batch_size)
    batched_val_dataset = val_dataset.batch(batch_size)

    #Build Iterator
    iterator = tf.contrib.data.Iterator.from_structure(batched_val_dataset.output_types, batched_val_dataset.output_shapes)
    images, labels = iterator.get_next()
    val_iterator_init_op = iterator.make_initializer(batched_val_dataset)

    # Build Model
    dropout_f = tf.placeholder("float")
    vgg = tf.contrib.slim.nets.vgg
    resnet_v1 = tf.contrib.slim.nets.resnet_v1
    image1 = images[:,0]
    image2 = images[:,1]

    if ws == True:
        with tf.variable_scope("net1") as scope:
            if net == VGG_NET:
                with slim.arg_scope(vgg.vgg_arg_scope(weight_decay=weight_decay)):
                    model1 , _ = vgg.vgg_16(image1, dropout_keep_prob=0.5, num_classes=last_layer_size)
            if net == RESNET50:
                with slim.arg_scope(resnet_v1.resnet_arg_scope()):
                    model1 , _ = resnet_v1.resnet_v1_50(image1, num_classes=last_layer_size)   
            scope.reuse_variables()
            if net == VGG_NET:
                with slim.arg_scope(vgg.vgg_arg_scope(weight_decay=weight_decay)):
                    model2 , _ = vgg.vgg_16(image2, dropout_keep_prob=0.5, num_classes=last_layer_size)
            if net == RESNET50:
                with slim.arg_scope(resnet_v1.resnet_arg_scope()):
                    model2 , _ = resnet_v1.resnet_v1_50(image2, num_classes=last_layer_size) 
    else:
        with tf.variable_scope("net1") as scope:
            if net == VGG_NET:
                with slim.arg_scope(vgg.vgg_arg_scope(weight_decay=weight_decay)):
                    model1 , _ = vgg.vgg_16(image1, dropout_keep_prob=0.5, num_classes=last_layer_size)
            if net == RESNET50:
                with slim.arg_scope(resnet_v1.resnet_arg_scope()):
                    model1 , _ = resnet_v1.resnet_v1_50(image1, num_classes=last_layer_size)    
        with tf.variable_scope("net2") as scope:
            if net == VGG_NET:
                with slim.arg_scope(vgg.vgg_arg_scope(weight_decay=weight_decay)):
                    model2 , _ = vgg.vgg_16(image2, dropout_keep_prob=0.5, num_classes=last_layer_size)
            if net == RESNET50:
                with slim.arg_scope(resnet_v1.resnet_arg_scope()):
                    model2 , _ = resnet_v1.resnet_v1_50(image2, num_classes=last_layer_size) 
    model1 = tf.reshape(model1 , [-1 , last_layer_size])
    model2 = tf.reshape(model2 , [-1 , last_layer_size])

    # Loss
    distance  = tf.sqrt(tf.reduce_sum(tf.pow(tf.subtract(model1, model2), 2), 1, keep_dims=True))
    distance = tf.cast(distance, tf.float32)
    # distance = tf.nn.l2_normalize(distance, 0, epsilon=1e-12, name=None)
    # distance = tf.div(tf.subtract(distance, tf.reduce_min(distance)), tf.subtract(tf.reduce_max(distance), tf.reduce_min(distance)))
    # distance =  distance * margin
    labels = tf.cast(labels, tf.float32)
    labels = tf.reshape(labels , [-1 , 1])
    loss = contrastive_loss(labels, distance)

    variables_to_restore = slim.get_variables_to_restore()
    tf.train.Saver(variables_to_restore)
    tf.train.Saver()
    
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.8
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())

        restorer = tf.train.import_meta_graph('output/model-20.meta')
        restorer.restore(sess, tf.train.latest_checkpoint('output/'))
                
        print('-------------------------------------Evaluating on Test Set')
        sess.run(val_iterator_init_op)
        predict = np.empty([0 , 1])
        gt = np.empty([0 , 1])
        while True:
            try:
                gt_tmp , predict_tmp = sess.run([labels , distance])
                predict = np.append(predict , predict_tmp)
                gt = np.append(gt , gt_tmp)
            except tf.errors.OutOfRangeError:
                break
        te_acc = compute_accuracy(predict, gt)
        te_loss = contrastive_loss_np(gt , predict)
        predict = predict.flatten()
        print predict
        print gt
        predict[predict < thresh] = 0.0
        predict[predict >= thresh] = 1.0
        print predict
        print('Accuracy test set %0.5f' % (te_acc*100)) 

def main(argv):
    usage_str = 'python train.py --mode=train/test'
    test_usage_str = 'python train.py --mode=test --image1=path/to/image1 --image2=path/to/image2 --model=path/to/model'

    mode = "train"
    image1 = None
    image2 = None
    model = None

    try:
        opts, args = getopt.getopt(argv, 'h', ['help=', 'mode=', 'image1=', 'image2=', 'model='])
    except getopt.GetoptError:
        print(usage_str)
        sys.exit(2)
    for opt, arg in opts:
        if opt in ('-h', '--help'):
            print(usage_str)
            sys.exit()
        elif opt == '--mode':
            mode = arg
        elif opt == '--image1':
            image1 = arg
        elif opt == '--image2':
            image2 = arg
        elif opt == '--model':
            model = arg

    if mode == "test":
        if not image1 or not image2 or not model:
            print('Please specify two images and model path. Usage:')
            print(test_usage_str)
            sys.exit(2)
        else:
            test(image1, image2, model)
    elif mode == "train":
        train()

if __name__ == '__main__':
    main(sys.argv[1:])