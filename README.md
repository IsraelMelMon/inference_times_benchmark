

SICARO: Deep Learning - Classification with TFLite
==================================================


Got a question? Send a message via Teams!
----------------------------------
If you are sure you've found a bug or an issue that can help the
code make less errors please contact via [email](imelendez@industriasrochin.com)
or teams to file a new issue:

What is SICARO: Deep Learning?
-------------

SICARO: deep learning, is a prototype intended for
real-time universal detection and classification of images in Python.  You should be able to 
use different trained models in the classification and be able to change HSV
thresholds for different color detection fruits and vegetables.

This code has only been developed for tomate grape (del Valle) and tested on 
MACOS and Ubuntu 20.04.

SICARO: deep learning is in development; some features are missing and there are bugs.
See 'Development status' below.

Requirements
------------

You need Python 3.5 or later to run the code.  You can have multiple Python
versions (2.x and 3.x) installed on the same system without problems.

In Ubuntu, Mint and Debian you can install Python 3 like this:

    $ sudo apt-get install python3 python3-pip

For other Linux flavors, macOS and Windows, packages are available at

  https://www.python.org/getit/


Quick start
-----------

Install requirements with pip:

    $ pip3 install -r requirements.txt


Running a classification only demo
---------------

Depending on your demo needs there are multiple demos that can help run this code
in different manners, in general you can run series inferences as:

    $ python3 inference_imgs.py --folder=[path to 2x resized crops] \
    --model_file=[path to tflite model file] --label_file=[path to class labels txt file] \
    --num_threads=[number of threads to use]
    
This should automatically time and print inference times over each cropped and resized image.


Running a detection and classification demo (needs GPU & code optimization)
---------------


In general, this is the pipeline to be used in the real-time detection. 
Batch inference still needs to be optimized to run parallel inference computations in a better time frame. 

    $ python3 create_bounding_boxes_classic.py --input=[path to primera_s_d or resaga_s_d] \
    --number=[limit number of images to process] --color=[1 if rgb images, 0 if grayscale] \
    --crop=[ 0 doesn't save any imgs, 1 saves bounding boxes crops, -1 saves full image with drawn rectangles]
    --inference=[1 does the inference, 0 doesn't do the inference]
    --model_file=[path to tflite model file] --label_file=[path to class labels txt file] 
    
by default one can also run:
    
    $ python3 create_bounding_boxes_classic.py 


This should automatically time and print inference times over each full input image and so, 
over each batch of tomato crops that corresponds to every input image (as in primera_s_d).



Development status
------------------

SICARO: deep learning is alpha software. Needs to refine number of flags, optimize parallel inferences,
use GPU for batch image processing, use real-time video frames and optimize code.

License
-------

SICARO: deep learning is licensed under the terms of RochinMX.
