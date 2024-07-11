import numpy as N
import matplotlib.pyplot as plt
#import tensorflow as tf
import math
import os
def sigmoid(x):
    return x if x>0 else 0
    # return x

def sigmoiderivative(x):
    return 1 if x>0 else 0
    # return 1

def Average(x):
    return N.exp(x)/sum(N.exp(x))
    # return 1

def load_mnist_images(filename):
    with open(filename, 'rb') as f:
        # Read magic number, number of images, rows, and columns
        magic_number = int.from_bytes(f.read(4), 'big')
        num_images = int.from_bytes(f.read(4), 'big')
        num_rows = int.from_bytes(f.read(4), 'big')
        num_cols = int.from_bytes(f.read(4), 'big')
        
        # Read all images
        raw_data = f.read()
        images = N.frombuffer(raw_data, dtype=N.uint8)
        images = images.reshape(num_images, num_rows * num_cols)
        
        return images / 255 # Normalize pixel values to [0, 1]
    
def load_mnist_labels(filename):
    with open(filename, 'rb') as f:
        # Read magic number, number of images, rows, and columns
        magic_number = int.from_bytes(f.read(4), 'big')
        num_labels = int.from_bytes(f.read(4), 'big')

        # Read all labels
        raw_data = f.read()
        labels = N.frombuffer(raw_data, dtype=N.uint8)

        return labels
# def forward_pass(input_data):
#     # Expand dimensions of input_data for batch processing
#     input_data_expanded = tf.expand_dims(input_data, axis=0)

#     # Assuming tf_W1, tf_W2, tf_W3, tf_b1, tf_b2, tf_b3 are defined elsewhere
#     # Example: tf_W1 = tf.ones((784, 16)), tf_b1 = tf.ones((16,))
    
#     # Transpose weight matrices if necessary
#     tf_W1_transposed = tf.transpose(tf_W1)
#     tf_W2_transposed = tf.transpose(tf_W2)
#     tf_W3_transposed = tf.transpose(tf_W3)

#     # First hidden layer with activation
#     hidden1_linear = tf.matmul(input_data_expanded, tf_W1_transposed) + tf_b1
#     hidden1 = tf.nn.sigmoid(hidden1_linear)

#     # Second hidden layer with activation
#     hidden2_linear = tf.matmul(hidden1, tf_W2_transposed) + tf_b2
#     hidden2 = tf.nn.sigmoid(hidden2_linear)

#     # Output layer for binary classification (sigmoid activation)
#     output_linear = tf.matmul(hidden2, tf_W3_transposed) + tf_b3
#     output = tf.nn.sigmoid(output_linear)

#     # Contract dimensions back to original shape
#     output_contracted = tf.squeeze(output, axis=0)
#     #print(output_contracted)
#     return output_contracted

    # Compute loss

# def compute_loss(predictions, targets):
#     # Ensure targets are cast to the same type as predictions
#     targets = tf.cast(targets, predictions.dtype)
    
#     # Expand dimensions of targets if necessary (assuming targets is [784])
#     targets_expanded = tf.expand_dims(targets, axis=0)  # Assuming axis=0 is correct for batch dimension
    
#     # Calculate squared error loss
#     loss = tf.reduce_mean(tf.square(predictions - targets_expanded))
    
#     return loss

def NN(Input,H1Weights,H1biases,NumWeights,Numbiases):
    H1output=[0]*16
    Number=[0]*10
    for i in range(16):
        for j in range(784):
            H1output[i]+=Input[j]*H1Weights[i][j]
        H1output[i]+=H1biases[i]
        Z1=H1output
        H1output[i]=sigmoid(H1output[i])
    for i in range(10):
        for j in range(16):
            Number[i]+=H1output[j]*NumWeights[i][j]
        Number[i]+=Numbiases[i]
        Z2=Number
    Number=Average(Number)
    return Number,H1output,Z1,Z2
def P(A2):
    return N.argmax(A2,0)
def A(Pr, Y):
    print(Pr,Y)
    return N.sum(Pr==Y) /Y.size
    
train_images_file = r"C:\Users\siddi\OneDrive\Desktop\mnist-master\mnist-master\train-images-idx3-ubyte\train-images.idx3-ubyte"
train_images = load_mnist_images(train_images_file)
#print(train_images[0])

train_labels_file = r"C:\Users\siddi\OneDrive\Desktop\mnist-master\mnist-master\train-labels-idx1-ubyte\train-labels.idx1-ubyte"
train_labels = load_mnist_labels(train_labels_file)
#print(train_labels[0:10])

# Random initialization of weights
H1Weights = N.random.randn(16, 784)  # Random values between 0 and 1 for a 16x784 matrix
NumWeights = N.random.randn(10, 16)  # Random values between 0 and 1 for a 10x16 matrix
H1biases = N.random.randn(16)        # Random values between 0 and 1 for a 16-element vector
Numbiases = N.random.randn(10)
P=N.ones(10)

#Automatic Gradient Calculation

# with open('parameters1.txt', 'r') as file:
#     lines = file.readlines()
    
#     # Read lines for H1Weights (from line 0 to 12543)
#     for line in lines[:12544]:
#         number = float(line.strip())
#         H1Weights.append(number)
#     H1Weights = N.array(H1Weights).reshape((16, 784))
    
#     # Read lines for H2Weights (from line 12544 to 12799)
#     for line in lines[12544:12800]:
#         number = float(line.strip())
#         H2Weights.append(number)
#     H2Weights = N.array(H2Weights).reshape((16, 16))
    
#     # Read lines for NumWeights (from line 12800 to 12959)
#     for line in lines[12800:12960]:
#         number = float(line.strip())
#         NumWeights.append(number)
#     NumWeights = N.array(NumWeights).reshape((10, 16))
    
#     # Read lines for H1biases (from line 12960 to 12975)
#     for line in lines[12960:12976]:
#         number = float(line.strip())
#         H1biases.append(number)
#     H1biases = N.array(H1biases)
    
#     # Read lines for H2biases (from line 12976 to 12991)
#     for line in lines[12976:12992]:
#         number = float(line.strip())
#         H2biases.append(number)
#     H2biases = N.array(H2biases)
    
#     # Read lines for Numbiases (from line 12992 to 13001)
#     for line in lines[12992:13002]:
#         number = float(line.strip())
#         Numbiases.append(number)
#     Numbiases = N.array(Numbiases)


#print(" Initial NumWeights: ",NumWeights,"Numbiases: ",Numbiases,"H2Weights:",H2Weights,"H2biases:",H2biases,"H1Weights: ",H1Weights,"H1biases: ",H1biases)

Input=[0.5]*784
Correction=[0.5]*10

for b in range(1000):
    Gradient_NumWeights=N.zeros((10,16))
    Gradient_Numbiases=[0]*10

    Gradient_H1Weights=N.zeros((16,784))
    Gradient_H1biases=[0]*16
    cost=0
    for Pic in range(1000):

        Input=train_images[Pic]
        Output,H1output,Z1,Z2=NN(Input,H1Weights,H1biases,NumWeights,Numbiases)
        Correction=[0]*10
        Correction[train_labels[Pic]]=1

        # #Automatic Gradient Calculation
        # tf_W1 = tf.Variable(H1Weights)
        # tf_b1 = tf.Variable(H1biases)
        # tf_W2 = tf.Variable(H2Weights)
        # tf_b2 = tf.Variable(H2biases)
        # tf_W3 = tf.Variable(NumWeights)
        # tf_b3 = tf.Variable(Numbiases)

        # with tf.GradientTape() as tape:
        #     predictions = forward_pass(Input)
        #     loss = compute_loss(predictions, Correction)

        # gradients = tape.gradient(loss, [tf_W1, tf_b1, tf_W2, tf_b2, tf_W3, tf_b3])

        # tf_gradients = [grad.numpy() for grad in gradients]
        # #print("Gradient for NumWeights:\n", tf_gradients[4])
        # #AutoGraddone
        Gradient_H1activation=[0]*16

        #OutputLayerstart
        for neuron in range(10):
            cost+=(Output[neuron]-Correction[neuron])**2
            for perneuron in range(16):
                Gradient_NumWeights[neuron][perneuron]+=2*(Output[neuron]-Correction[neuron])*H1output[perneuron]
            Gradient_Numbiases[neuron]+=2*(Output[neuron]-Correction[neuron])
        #For partial derivative of cost function wrt to activation of last Hidden layer
        for LastHiddenLayerNeuron in range(16):
            for neuron in range(10):
                Gradient_H1activation[LastHiddenLayerNeuron]+=2*(Output[neuron]-Correction[neuron])*NumWeights[neuron][LastHiddenLayerNeuron]
        #OutputlayerGradientparametersdone
        


        #First Hidden Layer
        for neuron in range(16):
            for perneuron in range(784):
                Gradient_H1Weights[neuron][perneuron]+=Gradient_H1activation[neuron]*sigmoiderivative(Z1[neuron])*Input[perneuron]
            Gradient_H1biases[neuron]+=Gradient_H1activation[neuron]*sigmoiderivative(Z1[neuron])
        #Gradient Complete... atleast in my brain, now to update values
    # print("Calculate Numw:",Gradient_NumWeights)
    # print(Z1[0])
    NumWeights = NumWeights - 0.0001*Gradient_NumWeights
    Numbiases = [x - 0.0001*y for x, y in zip(Numbiases, Gradient_Numbiases)]
    H1Weights = H1Weights - 0.0001*Gradient_H1Weights
    H1biases = [x - 0.0001*y for x,y in zip(H1biases,Gradient_H1biases)]
    if b%10==0:
        count=0
        for i in range(1000):
            Output,H1output,Z1,Z2=NN(train_images[i],H1Weights,H1biases,NumWeights,Numbiases)
            if max(Output)==Output[train_labels[i]]:
                count+=1
        print("Accuracy:",count/1000)
#---------------------ImageViewing
# image_index = 15

# # Access the image data
# image_data = train_images[image_index]

# # Reshape if needed (e.g., for grayscale images, reshape to 28x28)
# # Assuming your images are 28x28 pixels, adjust this according to your actual image size
# image_data = N.reshape(image_data, (28, 28))

# # Plotting
# plt.figure(figsize=(4, 4))  # Optional: Set the figure size
# plt.imshow(image_data, cmap='gray')  # Display the image in grayscale
# plt.title(f'Image {image_index}')  # Optional: Add a title
# plt.axis('off')  # Optional: Turn off axis labels
# plt.show()
# print("Answer",train_labels[image_index])
# file=open('M.text', 'w')
# H1Weights_str = "H1Weights = \n" + N.array2string(H1Weights, separator=', ', threshold=N.inf)
# H1biases_str = "H1biases = " + N.array2string(H1biases, separator=',')
# H2Weights_str = "H2Weights = " + N.array2string(H2Weights, separator=',')
# H2biases_str = "H2biases = " + N.array2string(H2biases, separator=',')
# NumWeights_str = "NumWeights = " + N.array2string(NumWeights, separator=',')
# Numbiases_str = "Numbiases = " + N.array2string(Numbiases, separator=',')

# # Writing to a file
# with open('network_parameters.txt', 'w') as file:
#     file.write(H1Weights_str + '\n')
#     file.write(H1biases_str + '\n')
#     file.write(H2Weights_str + '\n')
#     file.write(H2biases_str + '\n')
#     file.write(NumWeights_str + '\n')
#     file.write(Numbiases_str + '\n')