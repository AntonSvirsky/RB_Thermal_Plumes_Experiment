#################################################################
#                                                               #
#                        UTILITY FUNCTIONS                      #
#                                                               #
#################################################################


import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

def calib_click_event(points):
    def callback(event, x, y, flags, param):
        #used by create_prespective_mask
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(points) < 4:
                points.append((x, y))
                print(f"Point stored at ({x}, {y})")
            else:
                print(f"Cannot add more then four point. Press <Backspace> to remove")
                
    return callback



def order_points(a):
     #used by create_prespective_mask
     #sort points from top to bottom and left to right.
     
     # sort by y (top to bottom)
     idx_y = np.argsort(a[:, 0])
     a_sorted = a[idx_y]
 
     top = a_sorted[:2]
     bottom = a_sorted[2:]
 
     # sort each row by x (left to right)
     top = top[np.argsort(top[:, 1])]
     bottom = bottom[np.argsort(bottom[:, 1])]
 
     return np.vstack([top, bottom])
            
def get_prespective_points(img,dst_w,dst_h):
    #Create prespective mask
    points = []
    
    frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    cv2.namedWindow("Select corner points")
    cv2.setMouseCallback("Select corner points", calib_click_event(points))
    msg1_txt = '<L-Click> Add point. <Backspace> Remove last. <Esc> Abort.'
    H = frame_rgb.shape[0]
    while True:
        # Draw for visual feedback
        frame_copy = frame_rgb.copy()
        for pt in points:
            cv2.circle(frame_copy, pt, 5, (255, 0, 0), -1)
        
        if len(points)< 4:
            msg2_txt = f'Select corner points ({len(points)} / 4)'
        else:
            msg2_txt = 'Press <Enter> to accept'
            
        cv2.putText(frame_copy, msg1_txt, (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 1) 
        cv2.putText(frame_copy, msg2_txt, (0, H-20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 1)
        cv2.imshow("Select corner points", frame_copy)
         
        key = cv2.waitKey(1) & 0xFF
        if key == 13 and len(points)==4: # Enter
            break
        if key == 27:  # Escape
            return [], False
            
        if key == 8 and len(points)>0: #Backspace
            p = points.pop()
            print(f'Point {p} removed')
    
    src_pts = order_points(np.float32(points)) #order points
    
    print("Collected points (Top, Bottom, Left, Right):")
    print(src_pts)
    
    cv2.destroyWindow("Select corner points")
    #create prespective transform
    return src_pts, True # 


def apply_gamma_filter(frame, A, gamma):
    if gamma == 1:
        return np.clip(frame * A, 0, 255).astype(np.uint8)
    
    L_f = frame.astype(np.float32) / 255.0
    L_gamma = A*np.power(L_f, gamma)
    return np.clip(L_gamma * 255, 0, 255).astype(np.uint8)


def collect_point_click_event(points,mouse_pos):
    def callback(event, x, y, flags, param):
        #used by create_prespective_mask
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append(x)
            print(f"Point [{len(points)-1}] stored at x={x}")
            
        if event == cv2.EVENT_MOUSEMOVE:
            mouse_pos[0] = x
            mouse_pos[1] = y
    return callback

# Print iterations progress
def printProgressBar(iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Taken from: https://stackoverflow.com/questions/3173320/text-progress-bar-in-terminal-with-block-characters
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()
