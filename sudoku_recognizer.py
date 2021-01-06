import cv2 
import numpy as np
import tensorflow as tf 
from scipy import ndimage
import copy
from sudoku_solver import * 
from helper_funcs import *
import solver_extern


def sudoku_recognizer(img, model, last_grid):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(1,3),0)
    thresh  = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,13,3) #11, 11 ganz gut
    #cv2.imshow("Thresh1", thresh)
    #cv2.RETR_CCOMP statt cv2.RETR_TREE hat sich nicht bewährt
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE) #SIMPLE
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
    epsilon = 65 #65 ganz gut
    contour_areas = []

    for cnt in contours:
        approx = cv2.approxPolyDP(cnt,epsilon,True)
        if len(approx) == 4:
                contour_areas.append((cv2.contourArea(cnt), cnt, approx))

    if len(contour_areas) > 0:
        #biggest_area = max(contour_areas, key=lambda x: x[0])[1]
        points = max(contour_areas, key=lambda x: x[0])[2]
    else: 
        #biggest_area = contours[0]
        points = cv2.approxPolyDP(contours[0],epsilon,True)

    #Falls Rand um Sudoku gezeichnet werden soll
    #cv2.drawContours(img, biggest_area, -1, (0, 255, 0), 2)
    
    if len(points) == 4:
        points = points.reshape(4,2)
        A = points[0]
        B = points[1]
        C = points[2]
        D = points[3]

        AB = B - A
        AD = D - A
        BC = C - B
        CD = C - D

        eps_angle = 20
        eps_length = 1.25

        if lenghts_to_different(A,B,C,D,eps_length):
            return img

        if not (right_angle(get_angle(AB,AD),eps_angle) and 
                right_angle(get_angle(AB,BC),eps_angle) and
                right_angle(get_angle(BC,CD),eps_angle) and 
                right_angle(get_angle(AD,CD),eps_angle)):
            return img

        pts1 = order_points(points)
        pts2 = np.float32([[0,0],[504,0],[504,504],[0,504]])
        M = cv2.getPerspectiveTransform(pts1,pts2)
        dst = cv2.warpPerspective(img,M,(504,504)) 

        #cv2.imshow("DST1", dst)
        #Originales "gewarptes" Sudoku kopieren
        orginal_warp = np.copy(dst)

        dst_gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
        dst_blur = cv2.GaussianBlur(dst_gray,(1,3),0)
        dst_thresh  = cv2.adaptiveThreshold(dst_blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                            cv2.THRESH_BINARY,13,3) #11, 11 ganz gut
        sudoku = cv2.bitwise_not(dst_thresh)
        _, sudoku = cv2.threshold(sudoku, 150, 255, cv2.THRESH_BINARY)
        #cv2.imshow("test", sudoku)

    
    else:
        return img

    SIZE = 9

    #Initialize Sudoku Grid
    grid = []
    for i in range(SIZE):
        row = []
        for j in range(SIZE):
            row.append(0)
        grid.append(row)

    # Höhe und Breite der einzelnen Sudoku-Felder
    height = sudoku.shape[0]//SIZE
    width = sudoku.shape[1]//SIZE

    # Für jedes Feld 10 Prozent Rand
    edge_height = int(height*0.11)
    edge_width = int(width*0.11)

    for i in range(SIZE):
        for j in range(SIZE):
            crop_sudoku = sudoku[height*i+edge_height:height*(i+1)-edge_height,
                                    width*j+edge_width:width*(j+1)-edge_width]
                    
            crop_sudoku = cv2.bitwise_not(crop_sudoku)
            crop_sudoku = largest_component(crop_sudoku)

            digit_pic_size = 28
            crop_sudoku = cv2.resize(crop_sudoku, (digit_pic_size, digit_pic_size))

            if is_white(crop_sudoku):
                grid[i][j] = 0
                continue
            
            # Wenn hier, dann relativ sicher eine Zahl im Bild
            # Leere Felder werden gut erkannt
            _, crop_sudoku = cv2.threshold(crop_sudoku, 200, 255, cv2.THRESH_BINARY)
            crop_sudoku = crop_sudoku.astype(np.uint8)

            crop_sudoku = cv2.bitwise_not(crop_sudoku)
            shiftx, shifty = get_best_shift(crop_sudoku)
            shifted = shift(crop_sudoku, shiftx, shifty)
            crop_sudoku = shifted
            crop_sudoku = cv2.bitwise_not(crop_sudoku)
            ##Zur Überprüfung einzelne Felder anzeigen
            #cv2.imshow(str(i)+str(j), crop_sudoku)
            prediction = model([prepare(crop_sudoku)])
            grid[i][j] = np.argmax(prediction[0])+1
    
    # Kopie des noch nicht gelösten Sudokus
    user_grid = copy.deepcopy(grid)

    # #orginal_warp = write_solution_on_image(orginal_warp, grid, user_grid)

    # # Wenn das letzte Sudoku und das aktuelle Sudoku gleich sind,
    # # muss es nicht nochmal gelöst werden
    # #if np.array_equal(grid, last_grid) and last_grid:
    # if (not last_grid is None) and (contains_zero(last_grid)):
    #     print("Schleife 1")
    #     orginal_warp = write_solution_on_image(orginal_warp, last_grid, user_grid)
    #     result_sudoku = cv2.warpPerspective(orginal_warp, M, (img.shape[1], img.shape[0])
    #                                     , flags=cv2.WARP_INVERSE_MAP)
    # else:
    #     print("Schleife 2")
    #     # Mein Solver noch zu langsam, daher externer
    #     #solve(grid)
    #     #Ab hier ist grid gelöst
    #     solver_extern.solve_sudoku(grid)
    #     #grid = se.solve2(grid)
    #     orginal_warp = write_solution_on_image(orginal_warp, grid, user_grid)
    #     result_sudoku = cv2.warpPerspective(orginal_warp, M, (img.shape[1], img.shape[0])
    #                                         , flags=cv2.WARP_INVERSE_MAP)
    #     last_grid = copy.deepcopy(grid)

    solver_extern.solve_sudoku(grid)
    orginal_warp = write_solution_on_image(orginal_warp, grid, user_grid)
    result_sudoku = cv2.warpPerspective(orginal_warp, M, (img.shape[1], img.shape[0])
                                             , flags=cv2.WARP_INVERSE_MAP)                                        
    result = np.where(result_sudoku.sum(axis=-1, keepdims=True)!=0, result_sudoku, img)
    return result



