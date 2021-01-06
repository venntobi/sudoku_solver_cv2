import cv2 
import numpy as np
import tensorflow as tf 
import math 
from scipy import ndimage
import copy
from sudoku_solver import * 
from helper_funcs import *
import solver_extern
from sudoku_recognizer import *

#Bild der Kamera 
cap = cv2.VideoCapture(0)
digit_recognizer = tf.keras.models.load_model("mnist_model")


last_sudoku = None
while(True):
    ret, frame = cap.read() # Read the frame
    if ret == True:
        sudoku_frame = sudoku_recognizer(frame, digit_recognizer, last_sudoku) 
        cv2.imshow("Real Time Sudoku Solver", sudoku_frame) # Print the 'solved' image
        if cv2.waitKey(1) & 0xFF == ord('q'):   # Hit q if you want to stop the camera
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()