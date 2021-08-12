import numpy as np
import glob
import os, sys
import tensorflow as tf
from numpy import save
import cv2

import re

def atof(text):
    try:
        retval = float(text)
    except ValueError:
        retval = text
    return retval

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    float regex comes from https://stackoverflow.com/a/12643073/190597
    '''
    return [ atof(c) for c in re.split(r'[+-]?([0-9]+(?:[.][0-9]*)?|[.][0-9]+)', text) ]

def cnn():


    IMAGE_SIZE = [75, 150]

    model_path = 'inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'

    cnn = tf.keras.applications.inception_v3.InceptionV3(
        include_top=False, input_tensor=None, weights=model_path,
        input_shape=IMAGE_SIZE + [3], pooling=None, classes=2,
        classifier_activation='softmax',)
    # cnn.load_weights(model_path)
    for layer in cnn.layers:
        layer.trainable = False
    layer_output = cnn.get_layer("mixed10").output
    intermediate_model = tf.keras.models.Model(inputs=cnn.input, outputs=layer_output)

    return intermediate_model


def extract(class_type ="alert"):

    intermediate_model =cnn()

    currpath = os.getcwd()
    npyfilespath= os.path.join(currpath,'data', class_type)
    os.chdir(npyfilespath)
    npfiles= glob.glob("*.jpg")
    npfiles.sort(key = natural_keys)
    all_arrays=[]
    group =[]
    count =0
    for i, npfile in enumerate(npfiles):
        name = npfile.split("_")[0]
        path = os.path.join(npyfilespath, npfile)
        img = cv2.imread(path)
        croppedImg1 = img / 255
        test_image = np.expand_dims(croppedImg1, axis=0)

        if count % 50 != 0 or count == 0:

            intermediate_prediction = intermediate_model.predict(test_image)

            group.append(intermediate_prediction)
            count = count + 1

        else:
            array = np.concatenate(group)
            if class_type =='alert':
                npfile =f"{name}_{str(i)}"
            else:
                npfile = f"{name}_10_{str(i)}"
            savepath = os.path.join(currpath,"data","sequences",npfile)

            save(savepath, array)

            group = []
            intermediate_prediction = intermediate_model.predict(test_image)
            group.append(intermediate_prediction)
            count = count + 1
    array = np.concatenate(group)
    if class_type == 'alert':
        npfile = f"{name}_{str(i+1)}"
    else:
        npfile = f"{name}_10_{str(i+1)}"
    savepath = os.path.join(currpath, "data", "sequences", npfile)

    save(savepath, array)


extract("alert") # for alert images
extract("drowsy") # for drowsy images




