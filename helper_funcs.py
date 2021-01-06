import cv2 
import numpy as np
import tensorflow as tf 
import math 
from scipy import ndimage
import copy


def contains_zero(grid):
    return not np.array(grid).all()

def showImage(img, name, width, height):
    new_image = np.copy(img)
    new_image = cv2.resize(new_image, (width, height))
    cv2.imshow(name, new_image)

def get_best_shift(img):
  cy, cx = ndimage.measurements.center_of_mass(img)
  rows, cols = img.shape
  shiftx = np.round(cols/2.0-cx).astype(int)
  shifty = np.round(rows/2.0-cy).astype(int)

  return shiftx, shifty

def shift(img, sx, sy):
  rows, cols = img.shape
  M = np.float32([[1,0,sx],[0,1,sy]])
  shifted = cv2.warpAffine(img,M,(cols,rows))
  return shifted


#Funktion um die Vier Eckpunkte erkannter Formen richtig anzuordnen
def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    
    return rect

#Funktion, die überprüft ob übergebenes Bild weiß/leer ist
def is_white(img):
    #w wenn allgemein viele Weiße Pixel
    # oder wenn in der Mitte des Bildes mehr als 90% weiße Pixel
    
    if img.sum() >= img.shape[0]*img.shape[1]*255 - img.shape[0]*255:
        return True

    middle_x = img.shape[0]//2
    middle_y = img.shape[1]//2
    
    #pixel_steps x and y 
    psx = int(0.2*img.shape[0])
    psy = int(0.2*img.shape[1]) 
    x1 = middle_x - psx
    x2 = middle_x + psx
    y1 = middle_y - psy
    y2 = middle_y + psy
    return img[x1:x2, y1:y2].sum() > (x2-x1)*(y2-y1)*255 - 255

#Funktion, die die größte Komponente im Bild findet
def largest_component(img):
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity=8)
    # Find the largest non background component.
    # Note: range() starts from 1 since 0 is the background label.
    if len(stats[:, -1]) <= 1:
        return np.ones(img.shape)*255

    max_label, max_size = max([(i, stats[i, cv2.CC_STAT_AREA]) 
                                for i in range(1, nb_components)], key=lambda x: x[1])
    img[output == max_label] = 255
    return img

#Funktion, die Bild für digit_recognizer passend macht
def prepare(img):
    array = img.reshape(-1,28,28,1)
    array.astype("float32")
    return array/255

#Funktion 1, um zu prüfen, ob Sudoku gefunden wurde
# Wenn ja, sollten alle vier Seitenlängen etwa gleich sein
# reicht, wenn längste und kürzeste ungefähr gleich sind 
def lenghts_to_different(A, B, C, D, eps):
    AB = math.sqrt((A[0]-B[0])**2 + (A[1]-B[1])**2)
    AD = math.sqrt((A[0]-D[0])**2 + (A[1]-D[1])**2)
    BC = math.sqrt((B[0]-C[0])**2 + (B[1]-C[1])**2)
    CD = math.sqrt((C[0]-D[0])**2 + (C[1]-D[1])**2)

    shortest = min(AB, AD, BC, CD)
    longest = max(AB, AD, BC, CD)
    return longest/shortest > eps

#Funktion 2, um zu prüfen ob Sudoku gefunden wurde. 
# Alle winkel sollten etwa 90 grad haben 
def right_angle(angle, eps):
    return abs(angle-90) < eps

def get_angle(vector_1, vector_2):
    unit_vector_1 = vector_1 / np.linalg.norm(vector_1)
    unit_vector_2 = vector_2 / np.linalg.norm(vector_2)
    dot_product = np.dot(unit_vector_1, unit_vector_2)
    angle = np.arccos(dot_product)
    return angle*57.2958

def write_solution_on_image(image, grid, user_grid):
    # Write grid on image
    SIZE = 9
    width = image.shape[1] // 9
    height = image.shape[0] // 9
    for i in range(SIZE):
        for j in range(SIZE):
            if(user_grid[i][j] != 0):   
                continue                
            text = str(grid[i][j])
            off_set_x = width // 15
            off_set_y = height // 15
            font = cv2.FONT_HERSHEY_SIMPLEX
            (text_height, text_width), baseLine = cv2.getTextSize(text, font, fontScale=1, thickness=3)
            marginX = math.floor(width / 7)
            marginY = math.floor(height / 7)
        
            font_scale = 0.6 * min(width, height) / max(text_height, text_width)
            text_height *= font_scale
            text_width *= font_scale
            bottom_left_corner_x = width*j + math.floor((width - text_width) / 2) + off_set_x
            bottom_left_corner_y = height*(i+1) - math.floor((height - text_height) / 2) + off_set_y
            image = cv2.putText(image, text, (bottom_left_corner_x, bottom_left_corner_y), 
                                                  font, font_scale, (255,0,150), thickness=3, lineType=cv2.LINE_AA)
    return image
