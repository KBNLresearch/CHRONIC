#!/usr/bin/env python

import os
import shutil
import sys
import urllib
import warnings
import requests
import xml.etree.ElementTree as ET
import tensorflow as tf

from io import BytesIO

from PIL import Image
from flask import request, Flask, Response
from cv2.cv import *

FACE_TRASHOLD = 4

application = Flask(__name__)
application.debug = True

with tf.gfile.FastGFile("/var/www/imgclassify/retrained_graph.pb", 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')

label_lines = [line.rstrip() for line
               in tf.gfile.GFile("/var/www/imgclassify/retrained_labels.txt")]


def detectMisc(image_file):
    image_data = tf.gfile.FastGFile(image_file, 'rb').read()
    with tf.Session() as sess:
        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
        predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data})
        top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]

        score = 0

        for node_id in top_k:
            if score < predictions[0][node_id]:
                human_string = label_lines[node_id]
                score = predictions[0][node_id]
    if human_string == 'nomaps':
        human_string = 'unknown'

    return human_string

def detectFace(image_file):

    with open(image_file, 'rb') as fh:
        files = {'file': fh.read()}

    r = requests.post('http://ontw:5000', files=files)

    if r.text == 'face':
        return True

    '''
    image = LoadImage(image_file)
    grayscale = CreateImage((image.width, image.height), 8, 1)
    CvtColor(image, grayscale, CV_BGR2GRAY)

    storage = CreateMemStorage(0)
    EqualizeHist(grayscale, grayscale)
    cascade = Load('/var/www/imgclassify/haarcascade_frontalface_alt.xml')
    faces = HaarDetectObjects(grayscale, cascade, storage, 2, 2, CV_HAAR_DO_CANNY_PRUNING, (50,50))

    if faces:
        for f in faces:
            if f[1] > FACE_TRASHOLD:
                return True

    '''
    return False


def decode_image(img_name='ddd:010096851:mpeg21:p003:alto_4.jpg', save_path='/tmp/qsdasd.jpg'):
    #def parse_alto(url="http://resolver.kb.nl/resolve?urn=ddd:011001953:mpeg21:p001:alto"):
    #http://resolver.kb.nl/resolve?urn=ddd:011010260:mpeg21:p003:alto"):
    # tm - 1930

    RESOLVER = 'http://resolver.kb.nl/resolve?urn='
    url = RESOLVER + img_name.split('_')[0]
    img_nr = int(img_name.split('_')[1].split('.')[0])

    IMGVIEWER = 'http://imageviewer.kb.nl/ImagingService/imagingService?id=%s&x=%s&w=%s&y=%s&h=%s'

    data = requests.get(url)
    data = ET.fromstring(data.content)

    i = 0 #  Iterater for multiple images

    for item in data.iter():
        if item.tag.split('}')[1] == 'GraphicalElement':
            i+=1
            x = item.attrib.get('HPOS')
            w = item.attrib.get('WIDTH')
            y = item.attrib.get('VPOS')
            h = item.attrib.get('HEIGHT')

            src_image = url.replace(':alto', ':image')

            url1 = IMGVIEWER % (src_image.split('=')[1], x, w, y, h)
            src_image = requests.get(url1)
            if i == img_nr:
                im = Image.open(BytesIO(src_image.content))
                im.save(save_path)
                break

@application.route('/')
def index():
    image_file = request.args.get('src')
    if not image_file:
        return('Invoke with ?src=X for example ?src=local_path_to_file i.e. /opt3/KB1M/new_batch/1900/found/ddd:010126918:mpeg21:p005:alto_1.jpg')
    warnings.filterwarnings("ignore")
    #image_file = os.tmpnam() + '.jpg'
    #decode_image(src, image_file)

    return_str = ''

    if detectFace(image_file):
        return_str += 'face '
    #image_file="/opt3/KB1M/new_batch/1900/found/ddd:010126918:mpeg21:p005:alto_1.jpg"

    return_str += detectMisc(image_file)

    #try:
    #    os.unlink(image_file)
    #except:
    #    pass

    if return_str == 'face nomaps':
        return_str = 'face'

    return return_str.strip()

#if __name__ == '__main__':
#    application.run(port=8900)

#image_file="/opt3/KB1M/new_batch/1900/found/ddd:010126918:mpeg21:p005:alto_1.jpg"
#print(detectMisc(image_file))
