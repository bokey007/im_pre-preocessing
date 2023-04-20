# This file contains opencv processors wrapped arroud corresponding UI component

# We need to create a decorator that does :
#  1. wraps UI component
#  2. Time fuction execution
#  3. Displays input sahape and output shape  

from functools import wraps

def timer(orig_func):
    import time

    @wraps(orig_func)
    def wrapper(*args, **kwargs):
        t1 = time.time()
        result = orig_func(*args, **kwargs)
        t2 = time.time() - t1
        return (result, t2)

    return wrapper

# we will need two UI decorators 
# 1. TO DECORATE OUTSIDE
# 2. TO DECORATE FUCTIONS SPECIFIC ui SETTINGS
        # A. Setting name (description)
        # B. slider
        # C. radio


def processor_settings():
    @wraps(orig_func)
    def wrapper(*args, **kwargs):
        if args[1]:
            # subheade
            st.sidebar.subheader("Controls for {}".format(args[1]))
        if args[2]:
            # radio list
            thresh_type = st.radio('Specify thresholding type:', args[2], horizontal=True,)
        if args[2]:
            # slider list

        
        return orig_func(*args, **kwargs)

    return wrapper
    

def find_shape(orig_func):

    @wraps(orig_func)
    def wrapper(*args, **kwargs):
        t1 = time.time()
        result = orig_func(*args, **kwargs)
        t2 = time.time() - t1
        return (result, t2)

    return wrapper


# functions 
# INPT : Image, Data
# OPT : Image, Data

def map_color_space(imput_im, to_color_space):
    if color_space == "Gray scale":
        im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)

    elif color_space == "hsv":
        im = cv2.cvtColor(im, cv2.COLOR_RGB2HSV)

    elif color_space == "lab":
        im = cv2.cvtColor(im, cv2.COLOR_RGB2LAB)

    elif color_space == "brg":
        im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
        
    elif color_space == "ch_one":
        im, _, _ = cv2.split(im)

    elif color_space == "ch_two":
        _, im, _ = cv2.split(im)

    elif color_space == "ch_three":
        _, _, im = cv2.split(im)

    elif color_space == "merge_first_two_ch":
        im[:, :, 2] = np.zeros((im.shape[0], im.shape[1]))

    elif color_space == "merge_last_two_ch":
        im[:, :, 0] = np.zeros((im.shape[0], im.shape[1]))

    elif color_space == "merge_last_first_ch":
        im[:, :, 1] = np.zeros((im.shape[0], im.shape[1]))

    return im
