import io
import os
import glob

os.environ['THEANO_FLAGS']='cuda.root=/hpc/sw/cuda/7.5.18,floatX=float32,device=gpu1'

import lasagne
from lasagne.layers import InputLayer, DenseLayer, DropoutLayer
from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
from lasagne.layers import MaxPool2DLayer as PoolLayer
from lasagne.layers import LocalResponseNormalization2DLayer as NormLayer
from lasagne.utils import floatX

import skimage.transform
from skimage import transform, color

from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

import numpy as np
import matplotlib.pyplot as plt

import pickle

net = {}
net['input'] = InputLayer((None, 3, 224, 224))
net['conv1'] = ConvLayer(net['input'], num_filters=96, filter_size=7, stride=2, flip_filters=False)
net['norm1'] = NormLayer(net['conv1'], alpha=0.0001) # caffe has alpha = alpha * pool_size
net['pool1'] = PoolLayer(net['norm1'], pool_size=3, stride=3, ignore_border=False)
net['conv2'] = ConvLayer(net['pool1'], num_filters=256, filter_size=5, flip_filters=False)
net['pool2'] = PoolLayer(net['conv2'], pool_size=2, stride=2, ignore_border=False)
net['conv3'] = ConvLayer(net['pool2'], num_filters=512, filter_size=3, pad=1, flip_filters=False)
net['conv4'] = ConvLayer(net['conv3'], num_filters=512, filter_size=3, pad=1, flip_filters=False)
net['conv5'] = ConvLayer(net['conv4'], num_filters=512, filter_size=3, pad=1, flip_filters=False)
net['pool5'] = PoolLayer(net['conv5'], pool_size=3, stride=3, ignore_border=False)
net['fc6'] = DenseLayer(net['pool5'], num_units=4096)
net['drop6'] = DropoutLayer(net['fc6'], p=0.5)
net['fc7'] = DenseLayer(net['drop6'], num_units=4096)
net['drop7'] = DropoutLayer(net['fc7'], p=0.5)
net['fc8'] = DenseLayer(net['drop7'], num_units=1000, nonlinearity=lasagne.nonlinearities.softmax)
output_layer = net['fc8']

O = open('vgg_cnn_s.pkl','rb')
model = pickle.load(O)

list(model.keys())

CLASSES = model[b'synset words']
MEAN_IMAGE = model[b'mean image']

lasagne.layers.set_all_param_values(output_layer, model[b'values'])

def prep_raw_image(im):
    # Resize so smallest dim = 256, preserving aspect ratio
    h, w, _ = im.shape
    if h < w:
        im = skimage.transform.resize(im, (256, w*256//h), preserve_range=True)
    else:
        im = skimage.transform.resize(im, (h*256//w, 256), preserve_range=True)

    # Central crop to 224x224
    h, w, _ = im.shape
    im = im[h//2-112:h//2+112, w//2-112:w//2+112]

    rawim = np.copy(im).astype('uint8')

    # Shuffle axes to c01
    im = np.swapaxes(np.swapaxes(im, 1, 2), 0, 1)

    # Convert to BGR
    im = im[::-1, :, :]
    im = im - MEAN_IMAGE
    return rawim, floatX(im[np.newaxis])


photos = []
draws = []
unknown = []
for file in glob.glob("./TestImages/photos/*.jpg"):
    image = plt.imread(file)
    if(len(image.shape) == 2):
        image = color.gray2rgb(image)
	#photos.append(transform.resize(image, (400, 400)))
    photos.append(image) # photos = np.asarray(photos)


for file in glob.glob("./TestImages/draws/*.jpg"):
    image = plt.imread(file)
    if(len(image.shape) == 2):
        image = color.gray2rgb(image)
    draws.append(image)

for file in glob.glob("./TestImages/unknown/*.jpg"):
    image = plt.imread(file)
    if(len(image.shape) == 2):
        image = color.gray2rgb(image)
    unknown.append(image)

neural_photos = []
neural_draws = []
neural_unknown = []

nPhotos = len(photos)
nDraws = len(draws)
nUnknown = len(unknown)

for i in range(nPhotos):
    print(i)
    rawim, im = prep_raw_image(photos[i])
    encoding = np.array(lasagne.layers.get_output(net['pool2'], im, deterministic=True).eval())
    encoding = encoding.flatten()
    neural_photos.append(encoding)

for i in range(nDraws):
    print(i)
    rawim, im = prep_raw_image(draws[i])
    encoding = np.array(lasagne.layers.get_output(net['pool2'], im, deterministic=True).eval())
    encoding = encoding.flatten()
    neural_draws.append(encoding)

for i in range(nUnknown):
    print(i)
    rawim, im = prep_raw_image(unknown[i])
    encoding = np.array(lasagne.layers.get_output(net['pool2'], im, deterministic=True).eval())
    encoding = encoding.flatten()
    neural_unknown.append(encoding)

NEph = np.asarray(neural_photos)
NEdr = np.asarray(neural_draws)

'''
In any case, the standard SKlearn protocol is as follows:
## training data
SVM.fit(X_train,y_train)
## testing unseen data
y_predict = SVM.predict(X_test)

#then something like...
error = (y_predict - y_test).norm(2)

'''

clf = svm.LinearSVC(random_state=0)
X = np.concatenate((NEph, NEdr),axis=0)
y = np.concatenate((np.zeros( nPhotos ), np.ones( nDraws )),axis=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
clf.fit(X_train, y_train)


for count, filename in enumerate(glob.glob("./TestImages/unknown/*.jpg")):
    print("Classifying image: %s" % filename)
    y_predict = clf.predict(np.asarray(neural_unknown[count]))
    print(y_predict - y_test).norm(2)


'''

Error:


Classifying image: ./TestImages/unknown/ddd:010091792:mpeg21:p005:alto_1.jpg
Traceback (most recent call last):
  File "test.py", line 153, in <module>
    y_predict = clf.predict(np.asarray(neural_unknown[count]))
  File "/home/wfa010/.local/lib/python2.7/site-packages/sklearn/linear_model/base.py", line 324, in predict
    scores = self.decision_function(X)
  File "/home/wfa010/.local/lib/python2.7/site-packages/sklearn/linear_model/base.py", line 300, in decision_function
    X = check_array(X, accept_sparse='csr')
  File "/home/wfa010/.local/lib/python2.7/site-packages/sklearn/utils/validation.py", line 410, in check_array
    "if it contains a single sample.".format(array))
ValueError: Expected 2D array, got 1D array instead:
array=[ 0.          0.          0.         ...,  0.          1.45822346
  2.94218111].
Reshape your data either using array.reshape(-1, 1) if your data has a single feature or array.reshape(1, -1) if it contains a single sample.


'''
