# Imports
import matplotlib.pyplot as plt
import keras_ocr
import os
import cv2

def ocrText(file):
    # # get installed location
    # pytesseract.pytesseract.tesseract_cmd = './AoE-env/tesseract.exe'

    # # read image
    # img = cv2.imread(file)

    # # convert to gray scale
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # # performing OTSU threshold
    # ret, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)

    # # kernel size
    # rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 18))

    # # dilation application to threshold
    # dilation = cv2.dilate(thresh1, rect_kernel, iterations=1)

    # # finding contours
    # contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # # creating copy of image
    # im2 = img.copy()

    # # text file created and flushed
    # textFile = open("recognizedText.txt", "w+")
    # textFile.write("")
    # textFile.close()

    # # loop
    # for cnt in contours:
    #     x, y, w, h = cv2.boundingRect(cnt)

    #     # drawing rect on copied image
    #     rect = cv2.rectangle(im2, (x, y), (x + w, y + h), (0, 0, 255), 2)

    #     # cropping text block
    #     cropped = im2[y:y + h, x:x + w]

    #     # open file in append mode
    #     textFile = open("recognizedText.txt", "a")

    #     # apply OCR on cropped image
    #     text = pytesseract.image_to_string(cropped)

    #     # appending text to file
    #     textFile.write(text)
    #     textFile.write("\n")

    #     # close file
    #     textFile.close()

    #     # saving copied image to tempImg.png
    #     cv2.imwrite('./tempImg.png', im2)


    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    # pipeline
    pipeline = keras_ocr.pipeline.Pipeline()

    # images
    # images = [
    #     keras_ocr.tools.read(pics) for pics in [
    #         'C:\\Users\\aacke\\Desktop\\ValRed2022-Games\\Game2.jpg',
    #         'C:\\Users\\aacke\\Desktop\\ValRed2022-Games\\SPPV.jpg',
    #         'C:\\Users\\aacke\\Desktop\\ValRed2022-Games\\G1.jpg'
    #     ]
    # ]
    image = [keras_ocr.tools.read(file)]

    # predictions
    prediction_groups = pipeline.recognize(image)

    fig, axs = plt.subplots(nrows=len(image) + 1, figsize=(20, 20))
    for ax, image, predictions in zip(axs, image, prediction_groups):
        keras_ocr.tools.drawAnnotations(image=image, predictions=predictions, ax=ax)

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite('./tempImg.png', image)

    recText = []

    for i in prediction_groups:
        for j in i:
            recText.append(j[0])

    print(recText)

    return recText