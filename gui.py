import tkinter as tk
import tempObjDet as obj
import ocrTempDet as ocr
import numpy as np
import cv2
import pyautogui
from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk

# Global variables
global file
file = ''

# prompting the user to select a file
def browseFiles():
   global file
   file = filedialog.askopenfilename(
      initialdir="/", 
      title="Select a File", 
      filetypes=(
         ("all files", "*.*"),
         ("Text files", "*.txt*"),
         ("PNG files", "*.png*"),
         ("JPG files", "*.jpg*")))
   img = Image.open(file)
   img = img.resize((1250, 750))
   img.save('./tempImg.png')
   file = './tempImg.png'
   previewImage.config(file=file) 
    
# uses ocr to determine text in file
def readText():
   global text
   text = ocr.ocrText(file)
   textFile = './tempImg.png'
   previewImage.config(file=textFile)
   confidenceLabel.config(text="First words: " + text[0] + ", " + text[1] + ", " + text[3])
   root.update()

def readObj():
   global preds
   print("test")
   preds = obj.detectObj(file)
   # img = Image.open(obj.finalImg)
   # img.save('./tempImg.png')
   objFile = './tempImg.png'
   previewImage.config(file=objFile) 
   confidenceLabel.config(text=preds[0][0]+", "+ '{:.2f}'.format(preds[1][0]*100)+"% confidence\n"+
                           preds[0][1]+", "+ '{:.2f}'.format(preds[1][1]*100)+"% confidence\n"+
                           preds[0][2]+", "+ '{:.2f}'.format(preds[1][2]*100)+"% confidence")
   root.update()

def liveCapture():
   # specs for video
   resolution = (1920, 1080)
   codec = cv2.VideoWriter_fourcc(*"XVID")
   fps = 60.0

   # output file
   video = "Recording.avi"

   # video writer
   out = cv2.VideoWriter(video, codec, fps, resolution)

   # new empty window
   cv2.namedWindow("Live", cv2.WINDOW_NORMAL)
   cv2.resizeWindow("Live", 480, 270)

   while True:
      img = pyautogui.screenshot()

      # convert to numpy array and RGB
      outScreen = np.array(img)
      outScreen = cv2.cvtColor(outScreen, cv2.COLOR_BGR2RGB)

      # wrtie to output file
      out.write(outScreen)
      cv2.imshow('Live', outScreen)

      # stop recording when we press 'q'
      if cv2.waitKey(1) == ord('q'):
         break

   # release video writer
   out.release()

   # Destroy all windows
   cv2.destroyAllWindows()

# tkinter root initialization
root = Tk()

# Variables
textColor = 'black'
notReadyColor = 'grey'
outlineColor = 'blue'
backgroundColor = 'white'

# main window
root.title('Array of Engineers Demo')
root.geometry('1500x1000+200+200')
root.resizable(False, False)
root.config(bg=notReadyColor)

# frame initialization
# input frame
inputFrame = Frame(
   root,
   background=backgroundColor,
   highlightbackground=outlineColor,
   highlightcolor=outlineColor,
   highlightthickness=2
   )

# detection frame
detFrame = Frame(
   root,
   background=backgroundColor,
   highlightbackground=outlineColor,
   highlightcolor=outlineColor,
   highlightthickness=2
   )

# preview frame
preFrame = Frame(
   root,
   background=backgroundColor,
   highlightbackground=outlineColor,
   highlightcolor=outlineColor,
   highlightthickness=2
   )

# output frame
outFrame = Frame(
   root,
   background=backgroundColor,
   highlightbackground=outlineColor,
   highlightcolor=outlineColor,
   highlightthickness=2
   )

# column and row configuration
root.columnconfigure(0, weight=1)
root.columnconfigure(1, weight=3)
root.rowconfigure(0, weight=2)
root.rowconfigure(1, weight=2)

# image
previewImage = PhotoImage(file=file)

# file button
fileButton = Button(
   inputFrame,
   text="Choose Imgae/Video File",
   command=browseFiles,
)

recordButton = Button(
   inputFrame,
   text="Live",
   command=liveCapture,
)

# text button
textButton = Button(
   detFrame,
   text="Select Text",
   command=readText,
)

# shape button
shapeButton = Button(
   detFrame,
   text="Select Shape",
   background=notReadyColor
)

# color button
colorButton = Button(
   detFrame,
   text="Select Color",
   background=notReadyColor
)

# picture button
pictureButton = Button(
   detFrame,
   text="Select Picture",
   command=readObj
)

# labels
confidenceLabel = Label(outFrame, text="Confidence appears here", fg="black", bg="grey", width=25)
confidenceLabel2 = Label(outFrame, text="Confidence", fg="black")

optionsLabel = Label(detFrame, text="Detection Options", fg="black")

imageLabel = Label(preFrame, image=previewImage, text="Detection Preview", fg="black")
previewLabel = Label(preFrame, text="Preview", fg="black")

# fill out gui grid
previewLabel.grid(column=1, row=0, sticky=S, padx=5, pady=5, )
imageLabel.grid(column=1, row=1, padx=5, pady=5, )

optionsLabel.grid(column=0, row=1, sticky=S, padx=5, pady=5)

confidenceLabel2.grid(column=1, row=4, padx=5, pady=5, sticky=S)
confidenceLabel.grid(column=1, row=5, padx=5, pady=5, sticky=N)

textButton.grid(column=0, row=2, padx=5, pady=5)
shapeButton.grid(column=0, row=3, padx=5, pady=5)
colorButton.grid(column=0, row=4, padx=5, pady=5)
pictureButton.grid(column=0, row=5, padx=5, pady=5)
fileButton.grid(column=0, row=0, padx=5, pady=5)
recordButton.grid(column=0, row=1, padx=5, pady=5)

# Frame packing
inputFrame.grid(column=0, row=0, padx=5, pady=5)
detFrame.grid(column=0, row=3, padx=5, pady=5)
preFrame.grid(column=1, row=0, padx=5, pady=5)
outFrame.grid(column=1, row=3, padx=5, pady=5)

# main loop
root.mainloop()