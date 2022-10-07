# -*- coding: utf-8 -*-
"""
Created on Sun Sep 25 10:13:01 2022

@author: bokey
"""

class map_colore_space:
    def __init__(self, inpt_im):
        self.inpt_im = inpt_im
        self.im_dynamic = None 
        self.params = None

    def rgb_to_gray(self):
        self.im_dynamic = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return(self.im_dynamic)


