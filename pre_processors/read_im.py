import numpy as np
from  PIL import Image
import cv2

class read_im():
    def __init__(self, uploaded_file):
        self.uploaded_file = uploaded_file
        self.im_dynamic = None 
        self.params = None
    
    def upFile_PIL(self):
        self.im_dynamic = Image.open(self.uploaded_file)
        return self.im_dynamic

    def upFile_cv2(self):
        im = self.uploaded_file.getvalue()
        im = np.asarray(bytearray(im), dtype="uint8")
        self.im_dynamic = cv2.imdecode(im, cv2.IMREAD_COLOR)
        return self.im_dynamic

    def PIL_cv2():
        pass

    def cv2_PIL():
        pass