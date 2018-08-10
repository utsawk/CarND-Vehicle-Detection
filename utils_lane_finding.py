# importing packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
from moviepy.editor import VideoFileClip
from IPython.display import HTML

# source points
sx1, sy1, sx2, sy2, sx3, sy3, sx4, sy4 = 170, 720, 595, 450, 730, 450, 1150, 720
src = np.float32([[sx1, sy1],[sx2, sy2],[sx3, sy3],[sx4, sy4]]) 

# destination points
dx1, dy1, dx2, dy2, dx3, dy3, dx4, dy4 = 320, 720, 320, 0, 960, 0, 960, 720 
dst = np.float32([[dx1, dy1],[dx2, dy2],[dx3, dy3],[dx4, dy4]]) 

n = 3 # history of line function
# reset search after 10 frames
reset_search = 20

# for radius of curvature
ym_per_pix = 30/720 # meters per pixel in y dimension
xm_per_pix = 3.7/(dx4 - dx1) # meters per pixel in x dimension


def filter_color(image, lower, upper):
    """
    Applies a mask to isolate colors in HSV color space using lowe and upper tuples
    :param image: RGB image
    :param lower: tuple of lower mask
    :param upper: tuple of upper mask
    :return: image filtered with the mask
    :return: mask
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    image = cv2.bitwise_and(image, image, mask = mask)
    return image, mask


def draw_line_image(image, lines, color=[255, 0, 0], thickness=10):
    """
    returns an image with lines on them specified by lines
    :param image: RGB image
    :param lines: lines to be drawn
    :param color: line of color
    :thickness: thickness of line
    :returns: image with line drawn
    """
    line_image = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    for line in lines:
        cv2.line(line_image, (line[0], line[1]), (line[2], line[3]), color, thickness)
    return line_image

def weighted_image(image, initial_image, α=0.8, β=1., γ=0.):
    """
    The result image is computed as follows -> initial_img * α + img * β + γ
    NOTE: initial_img and img must be the same shape!
    :param image: image with lines drawn on them
    :param initial_img: original image without any processing
    :returns: weighted image initial_img * α + img * β + γ
    """
    return cv2.addWeighted(initial_image, α, image, β, γ)


# Color & gradient thresholding
def thresholding(image, r_thresh=(200, 255), s_thresh=(170, 255), sx_thresh=(20, 100)):
    """
    returns (R&S)|sobel-x where R is the R thresholded image (from RGB), 
                  S is the S thresholded image (from HLS), and 
                  sx_thresh is the sobel-x thresholded image
    :param image: RGB image
    :param r_thresh: R channel threshold
    :param s_thresh: S channel threshold
    :param sx_thresh: gradient x threshold
    """
    # Convert to HLS color space and separate the V channel
    r_channel = image[:,:,0]
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]
    # Sobel x
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0, ksize = 5) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    
    # Threshold R channel
    r_binary = np.zeros_like(r_channel)
    r_binary[(r_channel >= r_thresh[0]) & (r_channel <= r_thresh[1])] = 1
    
    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1
    
    # Threshold S channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    # Stack each channel
    color_binary = np.dstack((r_binary, sxbinary, s_binary)) * 255
    # Combine the two binary thresholds
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[((r_binary == 1) & (s_binary == 1)) | (sxbinary == 1)] = 1
    return r_binary, s_binary, sxbinary, color_binary, combined_binary


def warp(image, src, dst):
    """
    returns perspective transform of an image and a matrix used to invert the transform
    """
    image_size = (image.shape[1], image.shape[0])
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    warped_image = cv2.warpPerspective(image, M, image_size, flags=cv2.INTER_LINEAR)
    return warped_image, Minv

def sanity_check(lx, rx):
    """
    function returns True or False depending on if the lane width around the
    bottom of the image is within +/- margin of lane width
    """
    margin = 100
    lane_width = rx-lx
    if rx[-1]-lx[-1] < (dx4-dx1) - margin or rx[-1]-lx[-1] > (dx4-dx1) + margin:
        return False
    return True

def find_lane_pixels(binary_warped):
    """
    this function finds the lane pixels
    """
    offset = 200
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = offset + np.argmax(histogram[offset:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # HYPERPARAMETERS
    # Choose the number of sliding windows
    nwindows = 9
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0]//nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []
    
    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        
        # Draw the windows on the visualization image
        # cv2.rectangle(out_img,(win_xleft_low,win_y_low),
        # (win_xleft_high,win_y_high),(0,255,0), 2) 
        # cv2.rectangle(out_img,(win_xright_low,win_y_low),
        # (win_xright_high,win_y_high),(0,255,0), 2) 
        
        # Identify the nonzero pixels in x and y within the window #
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty

def polyfit_using_prev(binary_warped, left_fit, right_fit):
    """
    this function finds the lane pixels in the current frame using lane line pixels
    from the previous frame
    """
    # Assume you now have a new warped binary image 
    # from the next frame of video (also called "binary_warped")
    # It's now much easier to find line pixels!
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 100
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + 
    left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + 
    left_fit[1]*nonzeroy + left_fit[2] + margin))) 

    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + 
    right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + 
    right_fit[1]*nonzeroy + right_fit[2] + margin)))  

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    return leftx, lefty, rightx, righty

def eval_at_y(poly, y):
    """
    function evaluates a polynomial at y
    """
    return poly[0]*y**2 + poly[1]*y + poly[2]


# Line class to keep track of last n fits
class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False  
        # polynomial coefficients averaged over the last n iterations
        self.best_fit = None # return best fit based on last n fits
        self.last_n_fits = [] # list of last n fits
        self.frames_not_detected = 0 # keep track of number of undetected images
        self.radius_of_curvature = 0
    
    def calculate_fit(self, fit):
        """
        function that returns average of last 'n' polynomial fits
        :param fit: polynomial fit for current frame
        :returns: average of last 'n' fits
        """
        if self.detected:
            self.last_n_fits.append(fit)
            if len(self.last_n_fits) > n:
                self.last_n_fits.pop(0)
        else:
            self.frames_not_detected += 1     
        self.best_fit = np.average(self.last_n_fits, axis = 0)  
        return self.best_fit   




