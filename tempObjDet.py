import tensorflow_hub as hub
import tensorflow as tf
import cv2
import numpy
import pandas as pd

# Global variables

# Dimensions
width = 500
height = 400

def detectObj(image):
    # Load image
    img = cv2.imread(image)

    # Resizing to dimensions
    reImg = cv2.resize(img, (width, height))

    # Convert to rgb
    rgbImg = cv2.cvtColor(reImg, cv2.COLOR_BGR2RGB)

    # Converting to tensorflow in uint8
    rgbTensorImg = tf.convert_to_tensor(rgbImg, dtype=tf.uint8)

    # Adds dims to rgbTensorImg
    rgbTensorImg = tf.expand_dims(rgbTensorImg, 0)

    # Loading model directly from TendorFlow Hub
    imgDetector = hub.load("https://tfhub.dev/tensorflow/efficientdet/lite2/detection/1")

    # Loading csv with labels of classes
    labels = pd.read_csv('object_detection_model//labelmap.csv', sep=';', index_col='ID')
    labels = labels['OBJECT']

    # Creating prediction
    boxes, scores, classes, num_detections = imgDetector(rgbTensorImg)

    # Processing outputs
    outLabels = classes.numpy().astype('int')[0]
    outLabels = [labels[i] for i in outLabels]
    outBoxes = boxes.numpy()[0].astype('int')
    outScores = scores.numpy()[0]

    # Putting boxes and labels on image
    for score, (ymin, xmin, ymax, xmax), label in zip(outScores, outBoxes, outLabels):
        if score < 0.5:
            continue

        score = score * 100
        scoreTxt = f'{round(score)}%'
        imgBoxes = cv2.rectangle(rgbImg, (xmin, ymax), (xmax, ymin), (0, 0, 255), 2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(imgBoxes, label, (xmin, ymax - 10), font, 1, (255, 0, ), 2, cv2.LINE_AA)
        cv2.putText(imgBoxes, scoreTxt, (xmax, ymax - 10), font, 1, (255, 0, 0), 2, cv2.LINE_AA)

    rgbImg = cv2.cvtColor(rgbImg, cv2.COLOR_RGB2BGR)
    cv2.imwrite('./tempImg.png', rgbImg)
    
    print(outLabels)
    print(outScores)
    return outLabels, outScores