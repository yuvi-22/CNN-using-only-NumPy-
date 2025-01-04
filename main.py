import numpy as np 
import struct
import numpy as np
from image_retrival import images
from label_retirval import labels



def convolve(image, filters, bias, stride = 1, padding = 0):

    h, w= image.shape
    num_filter, _, filter_height, filter_width = filters.shape

    out_h = (h - filter_height) // stride  + 1
    out_w = (w - filter_width) // stride + 1

    feature_maps = np.zeros((num_filter, out_h, out_w))

    for f in range(num_filter):
        for i in range(0,out_h):
            for j in range(0, out_w):
                region = image[i*stride : i*stride + filter_height, j*stride : j*stride + filter_width]
                feature_maps[f,i,j] = np.sum(region * filters[f]) + bias[f]

    return feature_maps


def relu(x):
    return np.maximum(0, x)


def max_pool(feature_maps, pool_size = 2, stride = 2):
    n, h, w = feature_maps.shape

    out_h = (h-pool_size) // stride + 1
    out_w = (w-pool_size) // stride + 1

    pooled_maps = np.zeros((n, out_h, out_w))

    for f in range(n):
        for i in range(0, out_h):
            for j in range(0, out_w):
                region = feature_maps[f, i*stride : i*stride + pool_size, j*stride : j*stride + pool_size]
                pooled_maps[f,i,j] = np.max(region)

    return pooled_maps


def flatten(feature_maps):
    return feature_maps.ravel()


def fully_conntected(input_vector, weights, bias):
    return np.dot(weights, input_vector) + bias


def softmax(logits):
    exp_logits = np.exp(logits - np.max(logits))
    return exp_logits / np.sum(exp_logits)


def forward_pass(image, conv_weights, fc_weights, conv_bias, fc_bias):
    conv_outputs = convolve(image, conv_weights, conv_bias)

    print("Conv outputs shape:", conv_outputs.shape)
    print("Sample conv output (filter 0):", conv_outputs[0])

    relu_outputs = relu(conv_outputs)
    print("ReLU outputs shape:", relu_outputs.shape)
    print("Sample ReLU output (filter 0):", relu_outputs[0])

    pool_outputs = max_pool(relu_outputs)
    print("Pool outputs shape:", pool_outputs.shape)
    print("Sample pool output (filter 0):", pool_outputs[0])


    flattened_outputs = flatten(pool_outputs)
    print("Flattened outputs shape:", flattened_outputs.shape)
    print("Flattened outputs (sample):", flattened_outputs[:10])  # Display first 10 elements

    logits = fully_conntected(flattened_outputs, fc_weights, fc_bias)
    print("Logits:", logits)
    

    probabilites = softmax(logits)
    print("Probabilities:", probabilites)

    return probabilites 



def cross_entropy_loss(predictions, targets):
    return -np.sum(targets * np.log(predictions + 1e-9))/ targets.shape[0]



def backward_fc(d_out, input_vector, fc_weights):
    d_weights = np.dot(d_out[:, np.newaxis], input_vector[np.newaxis, :])  # Correctly align shapes
    d_bias = d_out
    d_input = np.dot(fc_weights.T, d_out)  # Backpropagate to the input of the FC layer

    return d_weights, d_bias, d_input


def backward_max_pool(d_pool, feature_maps, pool_size = 2, stride = 2):
    n,h,w = feature_maps.shape
    d_input = np.zeros_like(feature_maps)
    for f in range(n):
        for i in range(0,h,stride):
            for j in range(0, h, stride):
                region = feature_maps[f, i:i+pool_size, j:j+pool_size]
                max_i, max_j = np.unravel_index(np.argmax(region), region.shape)

                d_input[f, max_i + i, max_j + j] = d_pool[f, i//stride, j//stride]

    return d_input



def backward_convolve(d_conv, image,filters, stride = 1 ):
    n_f, _, filter_h, filter_w = filters.shape
    d_filters = np.zeros_like(filters)
    d_bias = np.zeros((n_f))
    d_input = np.zeros_like(image, dtype = np.float64)

    for f in range(n_f):
        for i in range(d_conv.shape[1]):
            for j in range(d_conv.shape[2]):

                region = image[i*stride : i*stride + filter_h, j*stride: j*stride + filter_w]

                d_filters[f] += d_conv[f, i, j] * region
                d_bias[f] += d_conv[f, i, j]
                d_input[i*stride : i*stride + filter_h, j*stride : j*stride + filter_w] += d_conv[f, i, j] * filters[f].squeeze()

    return d_input, d_filters, d_bias



def train(image, label, conv_weights, conv_bias, fc_weights, fc_bias, learning_rate = 0.001):
    conv_outputs = convolve(image, conv_weights, conv_bias)

    relu_outputs = relu(conv_outputs)
   
    pool_outputs = max_pool(relu_outputs)

    flattened_outputs = flatten(pool_outputs)
  
    logits = fully_conntected(flattened_outputs, fc_weights, fc_bias)
 
    probabilites = softmax(logits)

    one_hot_label = np.zeros((10))
    one_hot_label[label] = 1
    loss = cross_entropy_loss(probabilites, one_hot_label)

    d_logits = probabilites - one_hot_label
    d_fc_weights , d_fc_bias, d_flattened = backward_fc(d_logits,flattened_outputs , fc_weights)
    d_pool = d_flattened.reshape(pool_outputs.shape)
    d_relu = backward_max_pool(d_pool, relu_outputs)
    d_conv, d_filters, d_bias = backward_convolve(d_relu, image, conv_weights)

    return loss 






conv_weights = np.random.randn(1, 1, 3, 3) * np.sqrt(2 / 9)  # He initialization
conv_bias = np.zeros((1,))
fc_weights = np.random.randn(10, 1* 13 * 13)  
fc_bias = np.zeros((10,))


epochs = 5
learning_rate = 0.001
for epoch in range(epochs):
    total_loss = 0
    for i, (image , label) in enumerate(zip(images, labels)):
        loss = train(image, label, conv_weights, conv_bias, fc_weights, fc_bias, learning_rate)
        total_loss += loss
    print(f"Epoch {epoch + 1} loss = {total_loss/len(images)}")
    

