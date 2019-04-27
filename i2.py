from flask import Flask, render_template, request, redirect
from werkzeug.utils import secure_filename
import os
# from load import *
from skimage import color
import numpy
print (numpy.__version__)
import os
import sys
import json
import datetime
import numpy as np
import skimage.draw

# Root directory of the project
from mrcnn import visualize
# from mrcnn.visualize import save_image
from mrcnn.config import Config
from mrcnn import model as modellib, utils

ROOT_DIR = os.path.abspath("")
sys.path.append(ROOT_DIR)  # To find local version of the library
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

class damageConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "damage"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + damage

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9

def color_splash(image, mask):
    """Apply color splash effect.
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]

    Returns result image.
    """
    # Make a grayscale copy of the image. The grayscale copy still
    # has 3 RGB channels, though.
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
    # Copy color pixels from the original color image where mask is set
    if mask.shape[-1] > 0:
        # We're treating all instances as one, so collapse the mask into one layer
        mask = (np.sum(mask, -1, keepdims=True) >= 1)
        splash = np.where(mask, image, gray).astype(np.uint8)
    else:
        splash = gray.astype(np.uint8)
    return splash

class InferenceConfig(damageConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()
weights_path="mask_rcnn_damage_0010.h5"
model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=DEFAULT_LOGS_DIR)
model.load_weights(weights_path, by_name=True)





def detect_and_color_splash(model, image_path=None, video_path=None):
    assert image_path or video_path
    class_names = ['damage']
    # Image or video?
    if image_path:
        # Run model detection and generate the color splash effect
        print("Running on {}".format(image_path))
        # Read image
        image = skimage.io.imread(image_path)
        print ('aaa',image)

        # Detect objects
        r = model.detect([image], verbose=1)[0]
        print ('bbb',r)
        #######################APLLY HERE!!!##############################
        # splash = color_splash(image, r['masks'])

        gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
        # Copy color pixels from the original color image where mask is set
        mask=r['masks']
        if mask.shape[-1] > 0:
            # We're treating all instances as one, so collapse the mask into one layer
            mask = (np.sum(mask, -1, keepdims=True) >= 1)
            splash = np.where(mask, image, gray).astype(np.uint8)
        else:
            splash = gray.astype(np.uint8)


        image_name=os.path.basename(image_path)
        file_name = images_results+"\image_"+image_name+"_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
        visualize.display_instances(file_name,image, r['rois'], r['masks'], r['class_ids'],
                                    'damage', r['scores'])
        # Save output
        file_name =  images_results+"\image_"+image_name+"_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
        skimage.io.imsave(file_name, splash)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])
UPLOAD_FOLDER = os.path.basename('data')
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
H = 28
W = 28


images_sent_in='images_in'
images_results='image_results'

@app.route('/', methods=['GET'])
def index_page():
	return render_template('index2.html')


@app.route('/predict', methods=['POST'])
def predict():
	file = request.files['fileupload']
	filename = os.path.join(images_sent_in, file.filename)

	file.save(filename)
	detect_and_color_splash(model, image_path=filename)

    # response = np.array_str(np.argmax(out,axis=1))
    # return response
    # return render_template('predicted.html')

	# img = color.rgb2gray(imread(filename, mode='L'))
	# img = imresize(img, (28, 28))
	# img = img.reshape(1, 28, 28, 1)
	# with graph.as_default():
	# 	out = model.predict(img)
	# 	print(out)
	# 	print(np.argmax(out,axis=1))
	# 	response = np.array_str(np.argmax(out,axis=1))
	# 	return response
		#return render_template('predicted.html')
if __name__ == "__main__":
	app.run(host='0.0.0.0', port=80)#host='0.0.0.0', port=80
