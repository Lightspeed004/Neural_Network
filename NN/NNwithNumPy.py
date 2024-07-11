import numpy as np
def ReLU(x):
    return np.maximum(0, x)

def Average(x):
    return np.exp(x)/sum(np.exp(x))

def ReLUd(x):
    return x>0

def load_mnist_images(filename):
    with open(filename, 'rb') as f:
        # Read magic number, number of images, rows, and columns
        magic_number = int.from_bytes(f.read(4), 'big')
        num_images = int.from_bytes(f.read(4), 'big')
        num_rows = int.from_bytes(f.read(4), 'big')
        num_cols = int.from_bytes(f.read(4), 'big')
        
        # Read all images
        raw_data = f.read()
        images = np.frombuffer(raw_data, dtype=np.uint8)
        images = images.reshape(num_images, num_rows * num_cols)
        
        return images / 255 # Normalize pixel values to [0, 1]
    
def load_mnist_labels(filename):
    with open(filename, 'rb') as f:
        # Read magic number, number of images, rows, and columns
        magic_number = int.from_bytes(f.read(4), 'big')
        num_labels = int.from_bytes(f.read(4), 'big')

        # Read all labels
        raw_data = f.read()
        labels = np.frombuffer(raw_data, dtype=np.uint8)

        return labels
def init_Parameters():
    W1 = np.random.randn(16,784)
    B1 = np.random.randn(16,1)
    W2 = np.random.randn(10,16)
    B2 = np.random.randn(10,1)
    return W1,W2,B1,B2

def One_Hot(Y):
    O=np.zeros((Y.max()+1,Y.size))
    for i in range(0,Batchsize):
        O[Y[i]][i]=1
    return O

def Forward_Propagation(W1,B1,W2,B2,X):
    print(B1.shape)
    Z1 = W1.dot(X) + B1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + B2
    A2 = Average(Z2)

    return A2,Z2,A1,Z1
def Back_Propagation(X,W1,W2,A1,A2,Z1,Z2,Y):
    I=Y.size
    Correction=One_Hot(Y)
    D2=A2 - Correction 
    dW2=1/I * D2.dot(A1.T)   
    dB2=1/I * np.sum(D2,1)
    D1= W2.T.dot(D2) * ReLUd(Z1)
    dW1=1/I * D1.dot(X.T)   
    dB1=1/I * np.sum(D1,1) 
    return dW1, dW2, dB1,dB2
def Update(W1,dW1,W2,dW2,B1,dB1,B2,dB2,a):
    W1= W1 - a*dW1
    W2=W2 - a*dW2
    print(B1.shape,'1',dB1.shape)
    B1=B1 - a * np.expand_dims(dB1, axis=1)
    B2=B2 - a * np.expand_dims(dB2, axis=1)
    return W1,W2,B1,B2
def P(A2):
    return np.argmax(A2,0)
def A(Pr, Y):
    print(Pr,Y)
    return np.sum(Pr==Y) /Y.size

train_images_file = r"C:\Users\siddi\OneDrive\Desktop\mnist-master\mnist-master\train-images-idx3-ubyte\train-images.idx3-ubyte"
train_images = load_mnist_images(train_images_file)

train_labels_file = r"C:\Users\siddi\OneDrive\Desktop\mnist-master\mnist-master\train-labels-idx1-ubyte\train-labels.idx1-ubyte"
train_labels = load_mnist_labels(train_labels_file)
Batchsize=40000
train_images_transpose=train_images[:][0:Batchsize].T
train_labels=train_labels[0:Batchsize].T
print(train_labels.shape)
One_Hot(train_labels)

W1,W2,B1,B2=init_Parameters()
print(B1.shape)
Loops=1000
for i in range(Loops):
    print(B1.shape)
    A2,Z2,A1,Z1 = Forward_Propagation(W1,B1,W2,B2,train_images_transpose)
    dW1,dW2,dB1,dB2= Back_Propagation(train_images_transpose,W1,W2,A1,A2,Z1,Z2,train_labels)
    W1,W2,B1,B2=Update(W1,dW1,W2,dW2,B1,dB1,B2,dB2,0.7)
    if i%10==0:
        print("Loop:", i)
        print("Accuracy:", A(P(A2), train_labels))
print(W1,W2,B1,B2) 
