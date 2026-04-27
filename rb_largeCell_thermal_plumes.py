#################################################################
#                                                               #
#                        PARAMETERS                             #
#                                                               #
#################################################################
# Input parameters
vido_file    = 'rb_t40,0_b15,5_2.mp4'
vido_file    = 'rb_t30,0_b15,5_2.mp4'

video_dir    = "C:/Users/antonsv/OneDrive - Technion/Teaching/Lab 5/RB Large Cell cv/Movies/"
analysis_dir = "C:/Users/antonsv/OneDrive - Technion/Teaching/Lab 5/RB Large Cell cv/Analysis/"

# Output parameters
dst_w    = 600   # width (pixels) of output image
dst_h    = 600   # height (pixels) of output image
cs_list  = [50,] # x location (pixels) of cross-sections to record 
L_y      = 150   # height (cm) of the image (used to write dx on output images)  

# Runtime options
use_prev_prespective          = True # use previusly obtained prespective mask (if available)
use_prev_meanflow_img         = True # use previusly obtained reference image (if available)
manual_parameter_calibration  = True # Manualy adjust filter parameters during runtime
manual_cross_section          = True # Manualy select cross-sections during runtime
save_output_video_org         = True # Save output video of original video
save_output_video_flac        = True # Save output video of fluctuations

# Default filter parameters for fluctuations 
# refImage_gauss_R =  5 # Radius (pixels) of gaussian blur applied on reference image
# gamma_filt_amp   = 10 # Linear gain applied to flactuaion image
# gamma_filt_exp   =  1 # Exponential gain applied to flactuaion image
# diff_x_gauss_R   = 31 # Horizontal scale of gaussian blur applied to flactuations
# diff_y_gauss_R   =  1 # Vertical scale of gaussian blur applied to flactuations
# flac_color_map   = False # Use heat style map for fluctuations (only visual)

# Default filter parameters for treshold mask
# bg_sub    = True # Apply MOG algorythm for back ground subtractor 
# th_cutoff = 40   # [0-255] Cutoff value for mask 
# erod_k    = 1    # Size of the erosion kernel applied to tresholded image
# dilate_k  = 1    # Size of the dialation kernel applied to tresholded image


refImage_gauss_R = 3
gamma_filt_amp = 5
gamma_filt_exp = 0.9
diff_x_gauss_R = 5
diff_y_gauss_R = 5

bg_sub    = True
th_cutoff = 40
erod_k = 3
dilate_k = 3

flac_color_map   = False
#################################################################
#                                                               #
#                   CODE STARTS HERE                            #
#                                                               #
#################################################################
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import rb_largeCell_utils as utils
from datetime import datetime

## internal definitions
prespecive_transform_filename = 'prespective.npy'
mean_flow_filename = 'mean_flow.npy'
t_wait = 1

## Load video file
vidCap = cv2.VideoCapture(os.path.join(video_dir, vido_file))
org_width = int(vidCap.get(cv2.CAP_PROP_FRAME_WIDTH))
org_height = int(vidCap.get(cv2.CAP_PROP_FRAME_HEIGHT))
org_frame_count = int(vidCap.get(cv2.CAP_PROP_FRAME_COUNT))
org_fps = vidCap.get(cv2.CAP_PROP_FPS)

if not vidCap.isOpened():
    print("Error: Cannot open video file.")
    exit()

print(f'Loaded video: {vido_file}: {org_width}x{org_height}, {org_frame_count} frames @ {org_fps:.2f} fps')

## Create analuysis dir
folder_name = os.path.splitext(vido_file)[0] #take video name as folder name
out_dir = os.path.join(analysis_dir,folder_name) #output directory
out_dir_cs_img = os.path.join(out_dir,'cross-section imgs')
out_dir_cs_npy = os.path.join(out_dir,'cross-section data')

os.makedirs(out_dir, exist_ok=True) #make directory if it dose not exist
os.makedirs(out_dir_cs_img, exist_ok=True) #make directory if it dose not exist
os.makedirs(out_dir_cs_npy, exist_ok=True) #make directory if it dose not exist

## create output videos
if save_output_video_org:
    output_video_path = os.path.join(out_dir,'orig_video.avi')
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    videoWriter_org = cv2.VideoWriter(output_video_path,fourcc,int(org_fps), (dst_w,dst_h))  
if save_output_video_flac:
    output_video_path = os.path.join(out_dir,'flact_video.avi')
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    videoWriter_flac = cv2.VideoWriter(output_video_path,fourcc,int(org_fps), (dst_w,dst_h))  
    
def close_program(msg='Aborted'):
    # Combines all the steps to properly close the program
    print(msg)
    vidCap.release()
    if save_output_video_org:  videoWriter_org.release()
    if save_output_video_flac: videoWriter_flac.release()
    cv2.destroyAllWindows()
    sys.exit(0)

## Prespective transform
trans_path = os.path.join(out_dir, prespecive_transform_filename)
_, calibration_frame = vidCap.read() #get first frame

if use_prev_prespective:
    if os.path.isfile(trans_path):
        # if prespective file exists, load ancor points
        src_pts = np.load(trans_path)
        print('Loaded existing prespective transform')
    else:
        # Prespective file dose not exist
        print('Prespective transform not found')
        use_prev_prespective = False
        
if not use_prev_prespective:
    # generate ancor points for presepective from user input
    src_pts, sucsess = utils.get_prespective_points(calibration_frame,dst_w,dst_h)
    if not sucsess: close_program() # user abborted program 
    np.save(trans_path, src_pts)    # save ancor points
    
dst_pts = np.float32([ # Destantion points for prespective transform
    [0,0]     ,  [0, dst_w],  
    [dst_h, 0], [dst_h, dst_w]  
])    
prespTrans = cv2.getPerspectiveTransform(src_pts, dst_pts)

# Show prespective
calibration_frame_warped = cv2.warpPerspective(calibration_frame, prespTrans, (dst_w, dst_h))
info_text = '<Any key>: Accept. <Esc> Abort'
cv2.putText(calibration_frame_warped, info_text, (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)  
cv2.imshow('Prespective', calibration_frame_warped)
k = cv2.waitKey()
cv2.destroyWindow('Prespective')
if k == 27: #Esc
    close_program()
    
## Get mean flow image
mean_flow_img_path = os.path.join(out_dir, mean_flow_filename)
if use_prev_meanflow_img:
    #load referance image if file exists
    if os.path.isfile(mean_flow_img_path):
        ref_frame = np.load(mean_flow_img_path)
        print('Loaded existing mean flow image transform')
    else:
        print('Mean flow reference image missing')
        use_prev_meanflow_img = False

if not use_prev_meanflow_img:
    #create mean flow reference image by full movie average
    print('Generating mean flow reference image')
    vidCap.set(cv2.CAP_PROP_POS_FRAMES, 0) #reset to start of clip
    frame_sum   = np.zeros(shape = (dst_h,dst_w), dtype = np.int64);
    frame_count = 0;
    success, frame = vidCap.read()
    while success:
        frame = cv2.warpPerspective(frame, prespTrans, (dst_w, dst_h))
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        frame_sum   += gray_frame
        frame_count += 1
        
        success, frame = vidCap.read()
        
    avg_frame =  frame_sum/frame_count
    ref_frame =  avg_frame.astype(np.uint8)
    
    np.save(mean_flow_img_path,ref_frame)
    print('Mean flow reference created and saved')
    
    # show and save reference image
    plt.figure()
    plt.imshow(ref_frame)
    plt.title(f' {vido_file} \n Reference frame')
    plt.xlabel('x [pixels]')
    plt.ylabel('y [pixels]')
    plt.savefig(os.path.join(out_dir, 'reference frame.png'),bbox_inches='tight')
    plt.show()

## Ref frame postprocseeeing and filter definitions
ref_frame_filt = cv2.GaussianBlur(ref_frame, (refImage_gauss_R, refImage_gauss_R), 0)
dilate_kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_k,dilate_k))
erode_kernel   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erod_k,erod_k))
bg_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=True)

def diff_and_filters(gray_frame):
    #computes absolute diffrence from reference frame and applies filters
    #diff is the difrence between the reference and a snapshot
    diff = cv2.absdiff(gray_frame, ref_frame_filt)
    diff = utils.apply_gamma_filter(diff, gamma_filt_amp, gamma_filt_exp)
    diff = cv2.GaussianBlur(diff, (diff_x_gauss_R, diff_y_gauss_R), 0)
    return diff

def th_and_filters(gray_frame):
    #computes absolute diffrence from reference frame and applies filters
    frame_in = bg_subtractor.apply(gray_frame) if bg_sub else gray_frame
    _, thresh = cv2.threshold(frame_in, th_cutoff, 255, cv2.THRESH_BINARY)
    thresh = cv2.erode(thresh, erode_kernel, iterations=1)
    thresh = cv2.dilate(thresh, dilate_kernel, iterations=1) 
    return thresh

def get_output_frames(input_frame):
    #shift prespective and generate filtered images
    frame_warp = cv2.warpPerspective(input_frame, prespTrans, (dst_w, dst_h)) 
    gray_frame = cv2.cvtColor(frame_warp, cv2.COLOR_BGR2GRAY)
    diff   = diff_and_filters(gray_frame)
    thresh = th_and_filters(diff)
    return gray_frame,diff,thresh

## Function for parameter update from slidebars 
def update_refGauss(val):
    global refImage_gauss_R,ref_frame_filt
    refImage_gauss_R = 1 + 2*val
    ref_frame_filt =  cv2.GaussianBlur(ref_frame, (refImage_gauss_R, refImage_gauss_R), 0)
    
def update_filtHoriz(val):
    global diff_x_gauss_R
    diff_x_gauss_R = 1 + 2*val
    
def update_filtVert(val):
    global diff_y_gauss_R
    diff_y_gauss_R = 1 + 2*val
    
def update_diffAmp(val):
    global gamma_filt_amp
    gamma_filt_amp = val
    
def update_diffGamma(val):
    global gamma_filt_exp
    gamma_filt_exp = np.round(1+ val*0.1,1)
    
def update_cmap(val):
    global flac_color_map
    flac_color_map = bool(val)
    
def update_Erode(val):
    global erode_kernel, erod_k
    erod_k = 1 + 2*val
    erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erod_k,erod_k))

def update_Dilate(val):
    global dilate_kernel, dilate_k
    dilate_k = 1 + 2*val
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_k,dilate_k))

def update_Treshold(val):
    global th_cutoff
    th_cutoff = int(val)
    
def update_backGroundSub(val):
    global bg_sub
    bg_sub = bool(val)
    
## Live parameter calibration    
if manual_parameter_calibration:
    print('Manual parameter calibration. Press <Enter> to finish.')
    
    #Window for flactuations movie
    windowName1 = 'Parameter Calibration (diff+filters)'
    cv2.namedWindow(windowName1)
    cv2.createTrackbar('RefFilt' , windowName1, (refImage_gauss_R-1)//2     ,  10, update_refGauss)   # Slider for refImage_gauss_R
    cv2.createTrackbar('LinGain' , windowName1, gamma_filt_amp              , 100, update_diffAmp)    # Slider for gamma_filt_amp
    cv2.createTrackbar('ExpGain' , windowName1, int(10*(gamma_filt_exp-1))  ,  10, update_diffGamma)  # Slider for gamma_filt_exp
    cv2.setTrackbarMin('ExpGain' , windowName1, -10)                                                    
    cv2.createTrackbar('XGauss'  , windowName1, (diff_x_gauss_R-1)//2       ,  50, update_filtHoriz)  # Slider for diff_x_gauss_R
    cv2.createTrackbar('YGauss'  , windowName1, (diff_y_gauss_R-1)//2       ,  50, update_filtVert)   # Slider for diff_y_gauss_R
    cv2.createTrackbar('cmap'    , windowName1,  int(flac_color_map)        ,   1, update_cmap)       # Slider for flac_color_map
    
    #Window for treshold movie
    windowName2 = 'Parameter Calibration (treshold)'
    cv2.namedWindow(windowName2)
    cv2.createTrackbar('bgSub'      , windowName2, int(bg_sub)   ,   1, update_backGroundSub) # Slider for bg_sub
    cv2.createTrackbar('Treshold'   , windowName2, th_cutoff     , 255, update_Treshold) # Slider for th_cutoff
    cv2.createTrackbar('ErodeKer'   , windowName2, (erod_k-1)//2 ,  10, update_Erode)    # Slider for erod_k
    cv2.createTrackbar('DialateKer' , windowName2, (dilate_k-1)//2, 10, update_Dilate)   # Slider for dilate_k
    
    #Window for original movie
    windowName3 = 'Parameter Calibration (original)'
    cv2.namedWindow(windowName3)

    info_text = '<Enter>: Accept. <Esc> Abort' #Instruction text, siplayed on each image
    while manual_parameter_calibration:
        vidCap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        success, frame = vidCap.read()
        while success:
            #shift prespective and generate filtered images
            gray_frame,diff,thresh = get_output_frames(frame)
            
            # add text and show images
            diff_rgb = cv2.applyColorMap(diff, cv2.COLORMAP_JET) if flac_color_map else cv2.cvtColor(diff, cv2.COLOR_GRAY2RGB)
            th_rgb   = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)
            org_rgb = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2RGB)
            
            for img_rbg, wind in zip([diff_rgb, th_rgb, org_rgb], [windowName1,windowName2,windowName3]):
                cv2.putText(img_rbg, info_text, (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.imshow(wind, img_rbg)
            
            k = cv2.waitKey(t_wait) & 0xFF
            if k == 13:  # Enter - accept parameters and proceed
                manual_parameter_calibration = False
                break 
            if k == 27: # Escape - Abort and close
                close_program()
            success, frame = vidCap.read()
    print('Manual calibration complete:')
    print(f' # refImage_gauss_R = {refImage_gauss_R}')
    print(f' # gamma_filt_amp = {gamma_filt_amp}')
    print(f' # gamma_filt_exp = {gamma_filt_exp}')
    print(f' # diff_x_gauss_R = {diff_x_gauss_R}')
    print(f' # diff_y_gauss_R = {diff_y_gauss_R}')
    print(f' # erod_k = {erod_k}')
    print(f' # dilate_k = {dilate_k}')
    cv2.destroyAllWindows()


## Manual cross-section selection
if manual_cross_section:
    cross_section_points = cs_list;
    mouse_pos = [0,0];
    
    # create output windows
    csWindowName1 = "Select cross-sections (flactuations)"
    csWindowName2 = "Select cross-sections (treshold)"
    csWindowName3 = "Select cross-sections (original)"
    cs_windows = [csWindowName1,csWindowName2,csWindowName3]
    for wind in cs_windows:
        cv2.namedWindow(wind)
        cv2.setMouseCallback(wind, utils.collect_point_click_event(cross_section_points,mouse_pos))

    print('Selecting cross section ponts with <L-Click>. <Backspace> to undo')
    print('Press <Enter> when done.')
    
    while manual_cross_section:
        vidCap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        success, frame = vidCap.read()
        info_text = '<L-Click> Add new. <Backspace]> Undo. <Enter> Accept. <Esc> Abort.'
        
        while success:
            #shift prespective and generate filtered images
            gray_frame,diff,thresh = get_output_frames(frame)
            
            #add text, lines at cross-sections and show images
            frame_rgb  = cv2.cvtColor(gray_frame, cv2.COLOR_BGR2RGB)
            diff_rgb = cv2.applyColorMap(diff, cv2.COLORMAP_JET) if flac_color_map else cv2.cvtColor(diff, cv2.COLOR_GRAY2RGB)
            thresh_rdg = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)
            
            for img_rgb, wind in zip([diff_rgb, thresh_rdg,frame_rgb],cs_windows):
                for idx, pt_x in enumerate(cross_section_points): #add line at each cross-section
                    cv2.line(img_rgb,(pt_x,0),(pt_x,dst_h),(255,0,0),2)
                    cv2.putText(img_rgb, f"[{idx}]", (pt_x, dst_h//2), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)
                pos_text = f'(x,y) = {mouse_pos}'
                cv2.putText(img_rgb, info_text, (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1) 
                cv2.putText(img_rgb, pos_text, (0, dst_h-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255 , 0), 1)
                cv2.imshow(wind, img_rgb)
            
    
            k = cv2.waitKey(t_wait) & 0xFF
            if k == 13:  # Eneter - Accept selection
                manual_cross_section = False
                break 
            if k == 27: # Esc - Abort
                close_program()
            if k == 8 and len(cross_section_points)>0: #Backspace - undo 
                pt_del = cross_section_points.pop()
                print(f'Point x={pt_del} was removed')
            success, frame = vidCap.read()

    cs_list = cross_section_points
    cv2.destroyAllWindows()

## Fluctutation cross section and outuf movie
print('Obtaining cross-section plots')
print(f'Corss sections at x = {cs_list}')

heat_map     = np.zeros(shape = (len(cs_list), 3, dst_h,org_frame_count))
imgType_list = ['original', 'flactuations', 'treshold']

vidCap.set(cv2.CAP_PROP_POS_FRAMES, 0)
success, frame = vidCap.read()
i_t = 0
while success:
    #shift prespective and generate filtered images
    gray_frame,diff,thresh = get_output_frames(frame)
    
    if save_output_video_org:
        frame_rgb = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2RGB)
        videoWriter_org.write(frame_rgb)
    if save_output_video_flac:
        diff_rgb = cv2.applyColorMap(diff, cv2.COLORMAP_JET) if flac_color_map else cv2.cvtColor(diff, cv2.COLOR_GRAY2RGB)
        videoWriter_flac.write(diff_rgb)
    
    
    # collect all cross-sections 
    heat_map[:,0,:,i_t] = gray_frame[:,cs_list].T
    heat_map[:,1,:,i_t] = diff[:,cs_list].T
    heat_map[:,2,:,i_t] = thresh[:,cs_list].T
    i_t +=1
        
    success, frame = vidCap.read()
    
    if i_t % 50 == 0: #show progress bar
        utils.printProgressBar(i_t, org_frame_count, length = 20, prefix = 'Progress:')
utils.printProgressBar(org_frame_count, org_frame_count, length = 20, prefix = 'Progress:')   

## Create cross section plots
dt = (1/org_fps)*1E3
dx = L_y/dst_h
for i,p in enumerate(cs_list):
    for j,imgType in enumerate(imgType_list):
        
        plt.figure()
        plt.imshow(heat_map[i,j])
        plt.xlabel(f'Frame Number [dt = {dt:.2f}ms]')
        plt.ylabel(f'Position [dx = {dx:.2f}mm]')
        plt.title(f' {vido_file} \n {imgType} \n x = {p}')
        
        plt.savefig(os.path.join(out_dir_cs_img, f'cs_x={p}_{imgType}.png'),bbox_inches='tight')
        plt.show()
    
        np.save(os.path.join(out_dir_cs_npy, f'cs_x={p}_{imgType}'), heat_map[i,j])

## Write parameter file
parameter_file_path = os.path.join(out_dir, 'log.txt')
with open(parameter_file_path, "a") as f:
    f.write(f'{datetime.now()}')
    f.write(f'Input video: {vido_file}: {org_width}x{org_height}, {org_frame_count} frames @ {org_fps:.2f} fps \n')
    f.write(f'Output video: {dst_w}x{dst_h} \n')
    f.write('Filter parameters: \n')
    f.write(f' # refImage_gauss_R = {refImage_gauss_R} \n')
    f.write(f' # gamma_filt_amp = {gamma_filt_amp} \n')
    f.write(f' # gamma_filt_exp = {gamma_filt_exp} \n')
    f.write(f' # diff_x_gauss_R = {diff_x_gauss_R} \n')
    f.write(f' # diff_y_gauss_R = {diff_y_gauss_R} \n')
    f.write(f' # erod_k = {erod_k} \n')
    f.write(f' # dilate_k = {dilate_k} \n')
    f.write(f'Cross-sections: {cs_list} \n')
    f.write('----------------------   \n')
close_program(msg = 'Finished')