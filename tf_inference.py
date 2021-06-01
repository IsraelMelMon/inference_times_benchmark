

#model = tf.keras.models.load_model('rochin_model_grape_fine_tuned')
#model.summary()
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""label_image for tflite."""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse
import time

import numpy as np
from PIL import Image
import tensorflow as tf # TF2
import cv2


def load_labels(filename):
    with open(filename, 'r') as f:
        return [line.strip() for line in f.readlines()]

def lite_model(images):
    interpreter.allocate_tensors()
    interpreter.set_tensor(interpreter.get_input_details()[0]['index'], images)
    interpreter.invoke()
    return interpreter.get_tensor(interpreter.get_output_details()[0]['index'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    """
    parser.add_argument(
        '-i',
        '--image',
        default='/tmp/grace_hopper.bmp',
        help='image to be classified')

    """
    parser.add_argument(
        '-m',
        '--model_file',
        default='/tmp/mobilenet_v1_1.0_224_quant.tflite',
        help='.tflite model to be executed')
    parser.add_argument(
        '-l',
        '--label_file',
        default='/tmp/labels.txt',
        help='name of file containing labels')
    parser.add_argument(
        '--input_mean',
        default=0.0, type=float,
        help='input_mean')
    parser.add_argument(
        '--input_std',
        default=255.0, type=float,
        help='input standard deviation')
    parser.add_argument(
        '--num_threads', default=None, type=int, help='number of threads')
    parser.add_argument(
        '-f','--folder', type=str, help='nfolder of images to infer over')
    args = parser.parse_args()

    list_f = os.listdir(args.folder)
    list_f = [files for files in list_f if files!=".DS_Store"]

    interpreter = tf.lite.Interpreter(
    model_path=args.model_file, num_threads=8)

    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # check the type of the input tensor
    floating_model = input_details[0]['dtype'] == np.float32

    # NxHxWxC, H:1, W:2
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]
    result_confirm = []
    time_avg = []

    for img in list_f:
        img = Image.open(args.folder+img).resize((width, height))

        # add N dim
        input_data = np.expand_dims(img, axis=0)

        if floating_model:
            input_data = (np.float32(input_data) - args.input_mean) / args.input_std

        interpreter.set_tensor(input_details[0]['index'], input_data)

        start_time = time.time()
        interpreter.invoke()
        stop_time = time.time()

        output_data = interpreter.get_tensor(output_details[0]['index'])
        results = np.squeeze(output_data)
        print(results)
        if results[0] > results[1]:
            result_confirm.append(1)
        #top_k = results.argsort()[-2:][::-1]
        top_k = results.argsort()[:][::-1]
        labels = load_labels(args.label_file)
        for i in top_k:
            if floating_model:
                print('{:08.6f}: {}'.format(float(results[i]), labels[i]))
            else:
                print('{:08.6f}: {}'.format(float(results[i] / 255.0), labels[i]))

        time_ms = (stop_time - start_time) * 1000
        print('time: {:.3f}ms'.format(time_ms))

        time_avg.append(time_ms)
        

    print("Total errors: {}".format(sum(result_confirm)))
    print("Avg. time inference: {:.3f}ms".format(time_avg(sum)/len(time_avg)))
