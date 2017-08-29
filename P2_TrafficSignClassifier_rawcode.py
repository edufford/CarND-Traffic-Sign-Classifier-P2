""" Load pickled data """
import pickle
import csv

print('Loading dataset...')

training_file = './traffic-signs-data/train.p'
validation_file = './traffic-signs-data/valid.p'
testing_file = './traffic-signs-data/test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)

""" Build a dictionary for sign name classes """
sign_name_file = './signnames.csv'
sign_names = {}
with open(sign_name_file, mode='r') as f:
    reader = csv.reader(f)
    for row in reader:
        if row[0].isdigit():
            sign_names[int(row[0])] = row[1]

""" Build data sets """
X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']
print('Done.')


### Replace each question mark with the appropriate value. 
### Use python, pandas or numpy methods rather than hard coding the results

print('Summarizing dataset...')

# Number of training examples
n_train = len(X_train)

# Number of validation examples
n_validation = len(X_valid)

# Number of testing examples.
n_test = len(X_test)

# What's the shape of an traffic sign image?
image_shape = X_train[0].shape

# How many unique classes/labels there are in the dataset.
n_classes = len(set((y_train)))

print("Number of training examples =", n_train)
print("Number of validation examples =", n_validation)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)


### Data exploration visualization code goes here.
### Feel free to use as many code cells as needed.
import numpy as np
import matplotlib.pyplot as plt
import random

# Visualizations will be shown in the notebook.
#%matplotlib inline

print('Visualizing dataset...')

""" Make bar graph of sign class frequencies in training set """
fig, ax1 = plt.subplots(figsize=(15, 5))
bincount_train = np.bincount(y_train)
x_pos = np.arange(n_classes)
ax1.bar(x_pos, bincount_train, 0.7, color='k')
ax1.set_title('Number of images in each class for TRAINING set')
ax1.set_xticks(np.arange(0, n_classes))
ax1.set_ylabel('# of images')
plt.show()

""" Make bar graph of sign class frequencies in validation set """
fig, ax2 = plt.subplots(figsize=(15, 5))
bincount_valid = np.bincount(y_valid)
x_pos = np.arange(n_classes)
ax2.bar(x_pos, bincount_valid, 0.7, color='b')
ax2.set_title('Number of images in each class for VALIDATION set')
ax2.set_xticks(np.arange(0, n_classes))
ax2.set_ylabel('# of images')
plt.show()

""" Make bar graph of sign class frequencies in test set """
fig, ax3 = plt.subplots(figsize=(15, 5))
bincount_test = np.bincount(y_test)
x_pos = np.arange(n_classes)
ax3.bar(x_pos, bincount_test, 0.7, color='g')
ax3.set_title('Number of images in each class for TEST set')
ax3.set_xticks(np.arange(0, n_classes))
ax3.set_ylabel('# of images')
plt.show()

""" Print out dictionary of sign classes for reference """
for key, value in sorted(sign_names.items()):
    print('{} : {}'.format(key, value))
print()

""" Show multiple example images of all sign classes """
def get_example_image_idx(img_class, num):
    idx = np.where(y_train == img_class)
    img_idx = []
    for i in range(num):
        img_idx.append(random.choice(idx[0]))
    return img_idx
    
def plot_example_images(img_idx, image_data, colormap=None):
    num = len(img_idx)
    fig, ax = plt.subplots(1, num, figsize=(15, 10))
    for i in range(num):
        image = image_data[img_idx[i]]
        if colormap:
            ax[i].imshow(image, cmap = colormap)
        else:
            ax[i].imshow(image)                
        ax[i].axis('off')
    plt.subplots_adjust(wspace = 0)
    plt.show()

'''
print('Examples of each sign class\n')
for i in range(n_classes):
    print('{}: {}'.format(i, sign_names[i]))
    ex_idx = get_example_image_idx(i, 10)
    plot_example_images(ex_idx, X_train)
'''


### Preprocess the data here. It is required to normalize the data. Other preprocessing steps could include 
### converting to grayscale, etc.
### Feel free to use as many code cells as needed.

import cv2
from tqdm import tqdm

def normalize_image(image_data):
    """
    This function re-scales RGB images from [0, 255] to [-0.5, 0.5] by linear interpolation
    """
    image_norm = np.empty(shape=image_data.shape, dtype=np.float32)
    a, b = -0.5, 0.5
    x_min, x_max = 0, 255
    for idx_img in tqdm(range(image_data.shape[0]), desc='Normalize'):
        image_norm[idx_img] = ((image_data[idx_img] - x_min)/(x_max - x_min)*(b - a) + a)
    return image_norm


def adaptive_hist_HLS(image_data):
    """
    This function applies Contrast Limited Adaptive Histogram Equalization (CLAHE)
    to the HLS Lightness and Saturation channels of the images to improve visibility.
    The Hue channel is not modified to keep original coloring.
    See http://docs.opencv.org/3.1.0/d5/daf/tutorial_py_histogram_equalization.html
    """
    image_histcolor = np.empty(shape=image_data.shape, dtype=np.uint8)
    for idx_img in tqdm(range(image_data.shape[0]), desc='Equalize'):
        image_wk = np.copy(image_data[idx_img])
        image_wk = cv2.cvtColor(image_wk, cv2.COLOR_RGB2HLS)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(5,5))
        image_wk[:,:,1] = clahe.apply(image_wk[:,:,1])
        image_wk[:,:,2] = clahe.apply(image_wk[:,:,2])
        image_wk = cv2.cvtColor(image_wk, cv2.COLOR_HLS2RGB)
        image_histcolor[idx_img] = image_wk
    return image_histcolor


""" Pre-process images with adaptive histogram equalization and normalization """
X_train_hist = adaptive_hist_HLS(X_train)
X_train_hist_norm = normalize_image(X_train_hist)

X_valid_hist = adaptive_hist_HLS(X_valid)
X_valid_hist_norm = normalize_image(X_valid_hist)

X_test_hist = adaptive_hist_HLS(X_test)
X_test_hist_norm = normalize_image(X_test_hist)


""" Visualize example results """
print('Example of pre-processing:\n')
ex_idx = get_example_image_idx(24, 10) # 10 examples of sign class 0 

print('Original images')
plot_example_images(ex_idx, X_train)

print('Adaptive Histogram Equalization with HLS Color')
plot_example_images(ex_idx, X_train_hist)

print('Normalized [0, 255] -> [-0.5, 0.5]')
plot_example_images(ex_idx, X_train_hist_norm)


### Define your architecture here.
### Feel free to use as many code cells as needed.

import tensorflow as tf

def conv2d_relu(input_net, patch_size, input_depth, output_depth, strides=1, padding='VALID', mu=0, sigma=0.1):
    """
    2D Convolution Layer with truncated normal distribution weights and biases and RELU activation
    """
    W = tf.Variable(tf.truncated_normal([patch_size, patch_size, input_depth, output_depth], mean=mu, stddev=sigma))
    b = tf.Variable(tf.truncated_normal([output_depth], mean=mu, stddev=sigma))
    conv = tf.nn.conv2d(input_net, W, strides=[1, strides, strides, 1], padding=padding)
    conv = tf.nn.bias_add(conv, b)
    conv = tf.nn.relu(conv)
    return conv


def maxpool(input_net, k, padding='VALID'):
    """
    Max pooling layer with kernel size (k x k)
    """
    pool = tf.nn.max_pool(input_net, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding=padding)
    return pool
    
    
def fcl_relu(input_net, input_depth, output_depth, mu=0, sigma=0.1):
    """
    Fully Connected Layer with truncated normal distribution weights and biases and RELU activation
    """
    W = tf.Variable(tf.truncated_normal([input_depth, output_depth], mean=mu, stddev=sigma))
    b = tf.Variable(tf.truncated_normal([output_depth], mean=mu, stddev=sigma))
    fcl = tf.add(tf.matmul(input_net, W), b)
    fcl = tf.nn.relu(fcl)
    return fcl


def DuffNet(x, keep_prob):
    """
    Architecture based on LeNet-5 with increased feature maps and added multi-scale
    connection for 1st CNN+Pool layer.  Dropout is applied to the fully connected
    layers to reduce overfitting.  Learning rate decay is applied to stabilize learning
    accuracy curves.

    Layer 1: CNN1
                5x5 Convolution with RELU activation
                Input = 32 x 32 x 3
                Output = 28 x 28 x 16

    Layer 2: POOL1
                2x2 Max Pooling
                Input = 28 x 28 x 16
                Output = 14 x 14 x 16

    Layer 3: CNN2
                5x5 Convolution with RELU activation
                Input = 14 x 14 x 16
                Output = 10 x 10 x 32

    Layer 4: POOL2
                2x2 Max Pooling
                Input = 10 x 10 x 32
                Output = 5 x 5 x 32

    Layer 5: FCL1
                Fully Connected Layer with Multi-Scale connections and RELU activation
                Input = (14 x 14 x 16) + (5 x 5 x 32) = 3136 + 800 = 3936
                Output = 1024
                Dropout with 50% keep probability

    Layer 6: FCL2
                Fully Connected Layer with RELU activation
                Input = 1024
                Output = 1024
                Dropout with 50% keep probability
                
    Layer 7: FCL3
                Fully Connected Layer with RELU activation
                Input = 1024
                Output = 43
    """
    
    # Global access CNN layers for later visualization
    global conv1
    global conv2

    # CNN1: Input = 32x32x3. Output = 28x28x16. Activation = RELU.
    conv1 = conv2d_relu(input_net = x, patch_size = 5, input_depth = 3, output_depth = 16)
    print('conv1: {}'.format(conv1))

    # POOL1: Input = 28x28x16. Output = 14x14x16.
    pool1 = maxpool(conv1, 2)
    print('pool1: {}'.format(pool1))

    # CNN2: Input = 14x14x16. Output = 10x10x32. Activation = RELU.
    conv2 = conv2d_relu(input_net = pool1, patch_size = 5, input_depth = 16, output_depth = 32)
    print('conv2: {}'.format(conv2))

    # POOL2: Input = 10x10x32. Output = 5x5x32.
    pool2 = maxpool(conv2, 2)
    print('pool2: {}'.format(pool2))
   
    # Flatten POOL1: Input = 14x14x16. Output = 3136.
    flat_pool1 = tf.contrib.layers.flatten(pool1)
    print('flat_pool1: {}'.format(flat_pool1))

    # Flatten POOL2: Input = 5x5x32. Output = 800.
    flat_pool2 = tf.contrib.layers.flatten(pool2)
    print('flat_pool2: {}'.format(flat_pool2))
    
    # Combine POOL1+POOL2: Input = 3136+800. Output = 3936.
    flat = tf.concat(1, [flat_pool1, flat_pool2])
    print('flat: {}'.format(flat))
    
    # FCL1: Input = 3936. Output = 1024.
    fcl1 = fcl_relu(input_net = flat, input_depth = 3936, output_depth = 1024)
    print('fcl1: {}'.format(fcl1))

    # Dropout
    fcl1_drop = tf.nn.dropout(fcl1, keep_prob)
    print('fcl1_drop: {}'.format(fcl1_drop))
  
    # FCL2: Input = 1024. Output = 1024
    fcl2 = fcl_relu(input_net = fcl1_drop, input_depth = 1024, output_depth = 1024)
    print('fcl2: {}'.format(fcl2))
    
    # Dropout
    fcl2_drop = tf.nn.dropout(fcl2, keep_prob)
    print('fcl2_drop: {}'.format(fcl2_drop))
    
    #FCL3: Input = 1024. Output = 43.
    fcl3 = fcl_relu(input_net = fcl2_drop, input_depth = 1024, output_depth = n_classes)
    print('fcl3: {}'.format(fcl3))

    logits = fcl3
    return logits


""" Features and Labels """
x = tf.placeholder(tf.float32, (None, 32, 32, 3))
y = tf.placeholder(tf.int32, (None))
keep_prob = tf.placeholder(tf.float32)
one_hot_y = tf.one_hot(y, n_classes)

print("Model architecture defined.")


### Train your model here.
### Calculate and report the accuracy on the training and validation set.
### Once a final model architecture is selected, 
### the accuracy on the test set should be calculated and reported as well.
### Feel free to use as many code cells as needed.

from sklearn.utils import shuffle

""" Hyper parameters """
EPOCHS = 50
BATCH_SIZE = 256
LEARN_RATE_START = 0.001
LEARN_DECAY_ITER = 100
LEARN_DECAY_PCT = 0.96
LOG_BATCH_STEP = BATCH_SIZE * 100

""" Learning rate decay """
global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(LEARN_RATE_START, global_step,
                                           LEARN_DECAY_ITER, LEARN_DECAY_PCT, staircase=False)

""" Main network """
logits = DuffNet(x, keep_prob)

""" Tensor Flow model training operations """
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
training_operation = optimizer.minimize(loss_operation, global_step=global_step)

""" Tensor Flow model evaluation operations """
prediction = tf.nn.softmax(logits)
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: 1.0})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples


""" Copy training data for processing by neural network """
X_train_data = np.copy(X_train_hist_norm)
y_train_data = np.copy(y_train)

""" Copy validation data for processing by neural network """
X_valid_data = np.copy(X_valid_hist_norm)
y_valid_data = np.copy(y_valid)

""" Copy test data for processing by neural network """
X_test_data = np.copy(X_test_hist_norm)
y_test_data = np.copy(y_test)


""" Train the Model """
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train_data)
    
    # Measurements use for graphing loss and accuracy
    batches = []
    train_acc_batch = []
    valid_acc_batch = []

    # Start training cycles
    for i in range(EPOCHS):        
        X_train_data, y_train_data = shuffle(X_train_data, y_train_data)
        batches_pbar = tqdm(range(0, num_examples, BATCH_SIZE),
                            desc='Epoch {:>2}/{}'.format(i+1, EPOCHS), unit='batches')
        for offset in batches_pbar:
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train_data[offset:end], y_train_data[offset:end]
            
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: 0.5})
            
            # Log batches
            if not offset % LOG_BATCH_STEP:
                train_accuracy = evaluate(X_train_data, y_train_data)
                validation_accuracy = evaluate(X_valid_data, y_valid_data)
                previous_batch = batches[-1] if batches else 0
                batches.append(LOG_BATCH_STEP + previous_batch)
                train_acc_batch.append(train_accuracy*100)
                valid_acc_batch.append(validation_accuracy*100)
            
        print("Learning Rate = {:.8f}".format(learning_rate.eval()))
        
        train_accuracy = evaluate(X_train_data, y_train_data)
        print("Training Accuracy = {:.3f}%".format(train_accuracy*100))

        validation_accuracy = evaluate(X_valid_data, y_valid_data)
        print("Validation Accuracy = {:.3f}%".format(validation_accuracy*100))
        
    # Plot learning curves
    acc_plot = plt.subplot()
    acc_plot.plot(batches, train_acc_batch, 'b-', label='Training Accuracy')
    acc_plot.plot(batches, valid_acc_batch, 'r-', label='Validation Accuracy')
    acc_plot.set_xlabel('Image #')
    acc_plot.set_xlim([batches[0], batches[-1]])
    acc_plot.set_ylabel('Accuracy %')
    acc_plot.set_ylim([60, 100])
    acc_plot.legend(loc=4)
    acc_plot.grid(True)
    plt.show()
    
    # Save model result
    saver.save(sess, './duffnet')
    print("\nModel saved.")

    
""" Evaluate the model with Test data set """
with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))
    test_accuracy = evaluate(X_test_data, y_test_data)
    print("\nTest Accuracy = {:.3f}%".format(test_accuracy*100))


### Check frequency of mis-predicted test images and visualize some

N_EXAMPLES = 3

with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))
    test_predictions = sess.run(prediction, feed_dict={x: X_test_data, y: y_test_data, keep_prob: 1.0})
    test_correct = sess.run(correct_prediction, feed_dict={x: X_test_data, y: y_test_data, keep_prob: 1.0})
    miss_idx = np.where(test_correct == False)
    miss_idx = miss_idx[0]
    
    # Make bar graph of sign class frequencies in training set
    fig, ax1 = plt.subplots(figsize=(15, 5))
    bincount_train = np.bincount(y_train)
    x_pos = np.arange(n_classes)
    ax1.bar(x_pos, bincount_train, 0.7, color='k')
    ax1.set_title('Number of images in each class for TRAINING set')
    ax1.set_xticks(np.arange(0, n_classes))
    ax1.set_ylabel('# of images')
    plt.show()

    # Make bar graph of mis-predicted sign class frequencies in test set
    fig, ax2 = plt.subplots(figsize=(15, 5))
    bincount_train = np.bincount(y_test_data[miss_idx])
    x_pos = np.arange(n_classes)
    ax2.bar(x_pos, bincount_train, 0.7, color='r')
    ax2.set_title('Number of mis-predicted images in each class for TEST set')
    ax2.set_xticks(np.arange(0, n_classes))
    ax2.set_ylabel('# of images')
    plt.show()
    
    print("\n=====================================================================")
    for i in range(N_EXAMPLES):
        img_idx = miss_idx[i]
        plt.imshow(X_test[img_idx])
        plt.show()
        plt.imshow(X_test_hist[img_idx])
        plt.show()
        print("Predicted: {}".format(sign_names[test_predictions[img_idx].argmax()]))
        print("Actual: {}".format(sign_names[y_test_data[img_idx]]))
        print("\n=====================================================================")


### Load the images and plot them here.
### Feel free to use as many code cells as needed.

import matplotlib.image as mpimg

""" Define web image files and set their sign classes """
webimage_dir = "./web_signs/"
webimage_dict = {'websign01.jpg': 38,
                 'websign02.jpg': 25,
                 'websign03.jpg': 12,
                 'websign04.jpg': 36,
                 'websign05.jpg': 1}

""" Load and visualize web images """
print("Loading sign images from the web...")
X_web = np.empty(shape=(len(webimage_dict), 32, 32, 3), dtype=np.uint8)
y_web = np.empty(shape=(len(webimage_dict)), dtype=np.int32)

i = 0
for img_name, img_class in webimage_dict.items():
    image = mpimg.imread(webimage_dir + img_name)
    print('\n{}:'.format(sign_names[img_class]))
    plt.imshow(image)
    plt.show()
    X_web[i] = image
    y_web[i] = img_class
    i += 1

""" Pre-process web images """
X_web_hist = adaptive_hist_HLS(X_web)
X_web_hist_norm = normalize_image(X_web_hist)


### Run the predictions here and use the model to output the prediction for each image.
### Make sure to pre-process the images with the same pre-processing pipeline used earlier.
### Feel free to use as many code cells as needed.

""" Copy web image data for processing by neural network """
X_web_data = np.copy(X_web_hist_norm)
y_web_data = np.copy(y_web)

""" Evaluate the Model with Web images"""
with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))
    predicted_output = sess.run(prediction, feed_dict={x: X_web_data, y: y_web_data, keep_prob: 1.0})
    
    print("\n=====================================================================")
    for i in range(len(X_web)):
        plt.imshow(X_web[i])
        plt.show()
        print("Prediction = {}".format(sign_names[predicted_output[i].argmax()]))
        print("Actual = {}".format(sign_names[y_web[i]]))
        print("\n=====================================================================")


### Calculate the accuracy for these 5 new images. 
### For example, if the model predicted 1 out of 5 signs correctly, it's 20% accurate on these new images.

""" Calculate accuracy of web image predictions """
with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))
    test_accuracy = evaluate(X_web_data, y_web_data)
    print("Total Accuracy = {:.0f}%".format(test_accuracy*100))


### Print out the top five softmax probabilities for the predictions on the German traffic sign images found on the web. 
### Feel free to use as many code cells as needed.

from textwrap import wrap

""" Get top k=5 softmax probabilities for web images and visualize vs actual sign class """
with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))
    top_k_output = sess.run(tf.nn.top_k(predicted_output, k=5))
    
    print("\n=====================================================================")
    for i in range(len(top_k_output.values)):
        plt.imshow(X_web[i])
        plt.show()
        print("Actual: {}\n".format(sign_names[y_web[i]]))
        
        print("Predictions:")
        labels = []
        pred_values = []
        for j in range(len(top_k_output.values[i])):
            class_id = top_k_output.indices[i,j]
            labels.append(sign_names[class_id])
            confidence_val = top_k_output.values[i,j]*100
            pred_values.append(confidence_val)
            #print("{:.3f}% : {}".format(confidence_val, sign_names[class_id]))
           
        # Plot bar chart of top five softmax probabilities
        fig, ax = plt.subplots()
        pos = np.arange(5)
        width = 0.5
        labels = ['\n'.join(wrap(label, 20)) for label in labels] # wrap label text
        ax.barh(pos, top_k_output.values[i]*100, width, color='r', tick_label=labels)
        plt.gca().invert_yaxis()
        for i, val in enumerate(pred_values):
            ax.text(val + 1, i, "{:.3f}%".format(val)) # add data labels to bars
        plt.show()
        print("\n=====================================================================")


### Visualize your network's feature maps here.
### Feel free to use as many code cells as needed.

# image_input: the test image being fed into the network to produce the feature maps
# tf_activation: should be a tf variable name used during your training procedure that represents the calculated state of a specific weight layer
# activation_min/max: can be used to view the activation contrast in more detail, by default matplot sets min and max to the actual min and max values of the output
# plt_num: used to plot out multiple different weight feature map sets on the same block, just extend the plt number for each new feature map entry

def outputFeatureMap(image_input, tf_activation, activation_min=-1, activation_max=-1 ,plt_num=1):
    # Here make sure to preprocess your image_input in a way your network expects
    # with size, normalization, ect if needed
    # image_input =
    # Note: x should be the same name as your network's tensorflow data placeholder variable
    # If you get an error tf_activation is not defined it may be having trouble accessing the variable from inside a function
    activation = tf_activation.eval(session=sess,feed_dict={x : image_input})
    featuremaps = activation.shape[3]
    plt.figure(plt_num, figsize=(15,15))
    for featuremap in range(featuremaps):
        plt.subplot(10,8, featuremap+1) # sets the number of feature maps to show on each row and column
        plt.title('FeatureMap ' + str(featuremap)) # displays the feature map number
        if activation_min != -1 & activation_max != -1:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmin =activation_min, vmax=activation_max, cmap="gray")
        elif activation_max != -1:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmax=activation_max, cmap="gray")
        elif activation_min !=-1:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmin=activation_min, cmap="gray")
        else:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", cmap="gray")

""" Visualize CNN's for each web image """
for i in range(len(X_web)):
    print("\n=====================================================================")
    plt.imshow(X_web[i])
    plt.show()

    viz_image = np.empty(shape=(1, 32, 32, 3), dtype=np.float)
    viz_image[0] = X_web_data[i]
    plt.imshow(viz_image[0])
    plt.show()

    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint('.'))
    
        print("Features from 1st Convolution Activation")
        outputFeatureMap(viz_image, conv1, plt_num=16)
        plt.show()
    
        print("Features from 2nd Convolution Activation")
        outputFeatureMap(viz_image, conv2, plt_num=32)
        plt.show()
