from utils_vehicle_detection import *
from utils_lane_finding import *
import glob
import sys
    

def fit_polynomial(binary_warped, plot = False):
    """
    function fits a polynomial on a binary image
    """
    image_shape = binary_warped.shape
    left_curverad = left_line.radius_of_curvature
    right_curverad = right_line.radius_of_curvature
    # Find our lane pixels first
    if (not left_line.detected and left_line.frames_not_detected > \
                                                reset_search) or len(left_line.last_n_fits) == 0:
        leftx, lefty, rightx, righty = find_lane_pixels(binary_warped)
    else:
        leftx, lefty, rightx, righty = \
        polyfit_using_prev(binary_warped, left_line.last_n_fits[-1], right_line.last_n_fits[-1])
        # leftx, lefty, rightx, righty = find_lane_pixels(binary_warped)
    
    # Generate x and y values for plotting
    ploty = np.linspace(0, image_shape[0]-1, image_shape[0])
    if len(lefty) !=0 and len(righty) != 0:
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)
        left_curverad, right_curverad = measure_curvature_real(left_fit, right_fit, image_shape[0]-1)
        left_line.radius_of_curvature = left_curverad
        right_line.radius_of_curvature = right_curverad
    
        left_fitx = eval_at_y(left_fit, ploty)
        right_fitx = eval_at_y(right_fit, ploty)
        if not sanity_check(left_fitx, right_fitx):
            left_line.detected = False
            right_line.detected = False
        else:
            left_line.detected = True
            right_line.detected = True
    else:
        left_fit =[]
        right_fit = []
        left_line.detected = False
        right_line.detected = False
    
        
    if not plot:
        left_fit = left_line.calculate_fit(left_fit)
        right_fit = right_line.calculate_fit(right_fit)
    
    
    offset = 0
    # calculate offset from center of lane
    if len(left_fit) != 0:
        left_fitx = eval_at_y(left_fit, ploty)
        right_fitx = eval_at_y(right_fit, ploty)

        left_bottom_x = left_fitx[-1]
        right_bottom_x = right_fitx[-1]
        # offset calculation
        lane_midpoint = (left_bottom_x + right_bottom_x)/2.0
        camera_midpoint = (image_shape[1]-1)/2.0
        offset = (camera_midpoint - lane_midpoint)*xm_per_pix
    
    if plot:
        ## Visualization ##
        # Colors in the left and right lane regions
        # Create an output image to draw on and visualize the result
        # Create an output image to draw on and visualize the result
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))
        plt.figure(3)
        if not left_line.detected:
            print('Sanity check failed, displaying rejected lines')
        out_img[lefty, leftx] = [255, 0, 0]
        out_img[righty, rightx] = [0, 0, 255]

        # Plots the left and right polynomials on the lane lines
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')
        # plt.xlim(0, image_shape[1])
        # plt.ylim(image_shape[0], 0)
        plt.imshow(out_img)
        plt.title("Lane lines identified")
    return left_fitx, right_fitx, ploty, left_curverad, right_curverad, offset 

def measure_curvature_real(left_fit, right_fit, y_eval):
    """
    measures the curvature in meters
    """
    # Define conversions in x and y from pixels space to meters
    # Fit new polynomials to x,y in world space
    left_fit_cr = np.zeros_like(left_fit)
    left_fit_cr[0] = left_fit[0]*xm_per_pix/ym_per_pix**2
    left_fit_cr[1] = left_fit[1]*xm_per_pix/ym_per_pix
    left_fit_cr[2] = left_fit[2]*xm_per_pix
    
    right_fit_cr = np.zeros_like(right_fit)
    right_fit_cr[0] = right_fit[0]*xm_per_pix/ym_per_pix**2
    right_fit_cr[1] = right_fit[1]*xm_per_pix/ym_per_pix
    right_fit_cr[2] = right_fit[2]*xm_per_pix
    
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + \
                           left_fit_cr[1])**2)**1.5) / (2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + \
                            right_fit_cr[1])**2)**1.5) / (2*right_fit_cr[0])
    
    # Now our radius of curvature is in meters
    return(left_curverad, right_curverad)

def mark_lane_lines(undist, warped, ploty, left_fitx, right_fitx, Minv):
    """
    function returns an image with lane lines marked on input image
    """
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (warped.shape[1], warped.shape[0])) 
    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
    return result
    
def pipeline(image, plot = False):
    """
    pipeline for images, is also used to plot intermediate image outputs
    """
    # undistort image
    undistorted_image = undistort_image(image)
    
    # R&S|sobel-x thresholding
    _, _, _, _, threshold_image = thresholding(undistorted_image)
    
    # yellow mask
    _, mask_yellow = filter_color(undistorted_image, np.array([20,100,100]), np.array([50,255,255]))
    
    # white mask
    _, mask_white = filter_color(undistorted_image, np.array([0,0,220]), np.array([255,35,255]))
    
    # combine yellow and white mask
    mask = cv2.bitwise_or(mask_yellow, mask_white)
    comb_image = np.zeros_like(threshold_image)
    
    # combine mask and thresholded image
    comb_image[(mask > 0)&(threshold_image == 1)] = 1
    
    # warp the binary image
    warped_image, Minv = warp(comb_image, src, dst)
    if plot:
        plt.figure(2)
        plt.imshow(warped_image, cmap = "gray")
        plt.title("Binary warped image")
    
    # calculate polynomial fit
    left_fitx, right_fitx, ploty, left_curverad, right_curverad, offset = fit_polynomial(warped_image, plot)
    
    # superimpose lines on top of the polynomial
    superimposed_image = mark_lane_lines(undistorted_image, warped_image, ploty, left_fitx, right_fitx, Minv)
    cv2.putText(superimposed_image, "Left curvature = " + str(np.round(left_curverad, 2)) + " m",(40,40), \
                                        cv2.FONT_HERSHEY_PLAIN, 2, (255,0,0), 2)
    cv2.putText(superimposed_image,"Right curvature = " + str(np.round(right_curverad, 2)) + " m",(40,80), \
                                        cv2.FONT_HERSHEY_PLAIN, 2, (255,0,0), 2)
    cv2.putText(superimposed_image,"Offset = " + str(np.round(offset*100, 2)) + " cm",(40,120), \
                                        cv2.FONT_HERSHEY_PLAIN, 2, (255,0,0), 2)
    
    return superimposed_image


def find_vehicles(image):
    box_list_all = []
    for i in range(len(scale)):
        box_list = find_cars(image, ystart, ystop[i], scale[i], svc, X_scaler, 
                       orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)  

        box_list_all.extend(box_list)
        
    box_list_history.update_box_list(box_list_all)
    heat = np.zeros_like(image[:,:,0]).astype(np.float)
    
    # Add heat to each box in box list
    heat = add_heat(heat, box_list_history.bbox_list)

    # Apply threshold to help remove false positives
    heat = apply_threshold(heat, threshold)
    
    # Visualize the heatmap when displaying    
    heatmap = np.clip(heat, 0, 255)

    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    
    return labels

def find_lanes(undistorted_image):

    # R&S|sobel-x thresholding
    _, _, _, _, threshold_image = thresholding(undistorted_image)
    
    # yellow mask
    _, mask_yellow = filter_color(undistorted_image, np.array([20,100,100]), np.array([50,255,255]))
    
    # white mask
    _, mask_white = filter_color(undistorted_image, np.array([0,0,215]), np.array([255,40,255]))
    
    # combine yellow and white mask
    mask = cv2.bitwise_or(mask_yellow, mask_white)
    comb_image = np.zeros_like(threshold_image)
    
    # combine mask and thresholded image
    comb_image[(mask > 0)&(threshold_image == 1)] = 1
    
    # warp the binary image
    warped_image, Minv = warp(comb_image, src, dst)
    
    # calculate polynomial fit
    left_fitx, right_fitx, ploty, left_curverad, right_curverad, offset = fit_polynomial(warped_image)
    
    # superimpose lines on top of the polynomial
    superimposed_image = mark_lane_lines(undistorted_image, warped_image, ploty, left_fitx, right_fitx, Minv)
    cv2.putText(superimposed_image, "Left curvature = " + str(np.round(left_curverad, 2)) + " m",(40,40), \
                                        cv2.FONT_HERSHEY_PLAIN, 1, (255,0,0), 2)
    cv2.putText(superimposed_image,"Right curvature = " + str(np.round(right_curverad, 2)) + " m",(40,80), \
                                        cv2.FONT_HERSHEY_PLAIN, 1, (255,0,0), 2)
    cv2.putText(superimposed_image,"Offset = " + str(np.round(offset*100, 2)) + "cm",(40,120), \
                                        cv2.FONT_HERSHEY_PLAIN, 1, (255,0,0), 2)

    return superimposed_image


def pipeline(image):
    """
    pipeline for video, does not plot any intermediate images
    """
    # undistort image
    undistorted_image = undistort_image(image)
    superimposed_image = find_lanes(undistorted_image)
    labels = find_vehicles(undistorted_image)

    draw_img = draw_labeled_bboxes(superimposed_image, labels)

    
    return draw_img


def undistort_image(image):
    """
    undistort image
    :param image: original distorted image
    :returns undistorted image
    """
    return cv2.undistort(image, mtx, dist, None, mtx)

def camera_calibration():
    # Read in all the images into a list
    images = glob.glob('camera_cal/calibration*.jpg')

    # array to store object points and image points for all the images

    objpoints = [] # 3D object points in real world images
    imgpoints = [] # 2D points in image plane

    # prepare object points like (0,0,0), (1,0,0), etc.
    objp = np.zeros((9*6,3), np.float32)
    objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2) # x, y coordinates

    for file_name in images:
        # read file
        image = mpimg.imread(file_name)
        # convert to grayscale
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (9,6), None)
        
        # If corners are found, append object points and image points
        if ret:
            imgpoints.append(corners)
            objpoints.append(objp)

    # find camera calibration matrix and distortion coefficients        
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    return mtx, dist

if __name__ == "__main__":
    mtx, dist = camera_calibration()
    # load model
    try:
        data = joblib.load('model')
        svc = data['classifier']
        X_scaler = data['X_scaler']
    except FileNotFoundError:
        print("Oops, no trained model found. Please run train_svm.py first. Exiting.....")
        sys.exit()

    output = 'output.mp4'

    box_list_history = bounding_box(10)
    threshold = 4
    right_line = Line()
    left_line = Line()

    try:
        clip1 = VideoFileClip("project_video.mp4")
    except FileNotFoundError:
        print("Oops, no file named project_video.mp4. Exiting.....")
        sys.exit()
    white_clip = clip1.fl_image(pipeline) #NOTE: this function expects color images!!
    white_clip.write_videofile(output, audio=False)


    