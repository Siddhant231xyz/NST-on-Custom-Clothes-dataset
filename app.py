from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import tensorflow as tf
from tensorflow.keras.applications import vgg19
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

app = Flask(__name__)

# Configure upload folder and allowed extensions
UPLOAD_FOLDER = 'static/images'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    if 'sample_image' not in request.files or 'style_image' not in request.files:
        return redirect(request.url)

    sample_image = request.files['sample_image']
    style_image = request.files['style_image']
    iterations = int(request.form['iterations'])

    if sample_image.filename == '' or style_image.filename == '':
        return redirect(request.url)

    if sample_image and allowed_file(sample_image.filename) and style_image and allowed_file(style_image.filename):
        sample_filename = secure_filename(sample_image.filename)
        style_filename = secure_filename(style_image.filename)

        sample_path = os.path.join(app.config['UPLOAD_FOLDER'], sample_filename)
        style_path = os.path.join(app.config['UPLOAD_FOLDER'], style_filename)

        sample_image.save(sample_path)
        style_image.save(style_path)

        final_image_path = style_transfer(sample_path, style_path, iterations)

        return render_template('result.html', result_image=final_image_path)

    return redirect(request.url)

output_model_path = "C://Users//admin//Desktop//style_transfer_app//output//trained_model.pth"

def deprocess_image(x):
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype('uint8')
    return x

def content_loss(base_content, target):
    return tf.reduce_mean(tf.square(base_content - target))

def gram_matrix(input_tensor):
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
    return result/(num_locations)

def style_loss(base_style, gram_target):
    batch_size, height, width, channels = base_style.get_shape().as_list()
    gram_style = gram_matrix(base_style)
    return tf.reduce_mean(tf.square(gram_style - gram_target)) / (height * width * channels)

def style_transfer(sample_image_path, style_image_path, iterations):
    def preprocess_image(image_path):
        img = load_img(image_path, target_size=(224, 224))
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = vgg19.preprocess_input(img)
        return img

    content_image = preprocess_image(sample_image_path)
    style_image = preprocess_image(style_image_path)

    model = vgg19.VGG19(weights='imagenet', include_top=False)

    content_layer = 'block5_conv2'
    style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']

    outputs = [model.get_layer(name).output for name in style_layers + [content_layer]]
    style_model = Model(inputs=model.input, outputs=outputs)

    style_outputs = style_model(style_image)
    content_outputs = style_model(content_image)

    style_grams = [gram_matrix(style_feature) for style_feature in style_outputs]

    initial_image = tf.Variable(content_image, dtype=tf.float32)

    optimizer = tf.optimizers.Adam(learning_rate=5.0)

    for i in range(1, iterations + 1):
        with tf.GradientTape() as tape:
            output = style_model(initial_image)
            loss = sum([style_loss(output[j], style_grams[j]) for j in range(len(style_layers))])
            loss += content_loss(output[-1], content_outputs[-1])
        grad = tape.gradient(loss, initial_image)
        optimizer.apply_gradients([(grad, initial_image)])
        if i % 100 == 0:
            print(f'Iteration {i} completed')

    final_image = deprocess_image(initial_image.numpy()[0])

    cfg = get_cfg()
    cfg.merge_from_file("detectron2/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    cfg.DATASETS.TRAIN = ("dataset_train",)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = output_model_path
    cfg.SOLVER.BASE_LR = 0.001  # Adjust the learning rate based on your dataset and model architecture
    cfg.SOLVER.MOMENTUM = 0.9  # Momentum for SGD optimizer
    cfg.SOLVER.WEIGHT_DECAY = 0.0001  # Weight decay for regularization
    cfg.SOLVER.GAMMA = 0.1  # Multiplicative factor for reducing learning rate
    cfg.SOLVER.STEPS = (500, 800)  # Decrease the learning rate at these steps
    cfg.SOLVER.MAX_ITER = 1000  # Total number of iterations for training
    cfg.SOLVER.IMS_PER_BATCH = 2  # Batch size

    cfg.SOLVER.STEPS = []  # You can add learning rate steps here
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128  # You can adjust this
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 5  # Updated number of classes

    # Enable mask prediction
    cfg.MODEL.MASK_ON = True


    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    predictor = DefaultPredictor(cfg)

    outputs = predictor(final_image)

    instances = outputs["instances"].to("cpu")

    if "pred_masks" not in instances.get_fields():
        print("Model does not predict masks!")
    else:
        masks = instances.pred_masks.numpy()
        background_color = (255, 255, 255)  # White color
        background = np.ones(final_image.shape, dtype=np.uint8) * background_color
        for mask in masks:
            mask = mask.astype(np.uint8)
            blended_image = background.copy()
            blended_image[mask == 1] = final_image[mask == 1]
            background = blended_image

        background_uint8 = background.astype(np.uint8)

        


        sample_img = cv2.imread(sample_image_path)
        final_image_resized = cv2.resize(final_image, (sample_img.shape[1], sample_img.shape[0]))

        final_image_bgr = final_image_resized[:, :, ::-1]

        # Resize background to match final_image_bgr
        background_resized = cv2.resize(background_uint8, (final_image_bgr.shape[1], final_image_bgr.shape[0]))

        # Overlay styled content on resized background
        final_result = cv2.add(final_image_bgr, background_resized)

        output_image_path = "./static/images/output_image.jpg"
        cv2.imwrite(output_image_path, final_result)

    return output_image_path

if __name__ == "__main__":
    app.run(debug=True)
