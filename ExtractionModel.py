import numpy as np
import segmentation_models as sm
import cv2
from enum import Enum, auto
sm.set_framework('tf.keras')

class ExtractMode(Enum):
    LABEL = 1
    HANDLING_UNIT = 2
        
class ExtractionModel():
    def __init__(self):
        self.model_img_size = (512, 512)
        self.model = sm.Unet('efficientnetb3', classes=1, activation='sigmoid')
        self.preprocessor = sm.get_preprocessing('efficientnetb3')
        
    def set_mode(self, mode):
        if mode == ExtractMode.HANDLING_UNIT:
            model_file = 'best_model_handling_unit.h5'
            self.model.load_weights(model_file)
        elif mode == ExtractMode.LABEL:
            model_file = 'arcbest_final_label.h5'
            self.model.load_weights(model_file)
        else:
            print(f'{mode} is not a supported Extract Model mode')
        
    def get_mask(self, img):
        h, w = img.shape[:2]
        image = cv2.resize(img, self.model_img_size, interpolation = cv2.INTER_AREA)
        image = self.preprocessor(image)
        
        mask = self.model.predict(np.expand_dims(image, axis=0)).round().squeeze()
        mask = cv2.resize(mask, (w, h), interpolation = cv2.INTER_AREA)
        
        mask = self.isolate_largest_component(mask)
        
        return mask.astype(np.uint8)
    
    def crop(self, img):
        mask = self.get_mask(img)
        rect = self.get_contour_rect(mask)
        
        if rect is None:
            return mask
        x, y, w, h = rect
        
        return img[y:y+h, x:x+w, :]
    
    def extract(self, img):
        mask = self.get_mask(img)
        rect = self.best_component(mask)
        if rect is None:
            return mask, None
        box = cv2.boxPoints(rect)
        box = np.int0(box).astype(np.uint32)

        width = int(rect[1][0])
        height = int(rect[1][1])

        src_pts = box.astype("float32")
        dst_pts = np.array([[0, height-1],
                            [0, 0],
                            [width-1, 0],
                            [width-1, height-1]], dtype="float32")
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped = cv2.warpPerspective(img, M, (width, height))
#         warped = cv2.rotate(warped, cv2.cv2.ROTATE_90_CLOCKWISE)

        return mask, warped

    def best_component(self, img):
        result = img.copy().astype(np.uint8)
        contours, _ = cv2.findContours(result, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            return None
        c = max(contours, key = cv2.contourArea)
        if cv2.contourArea(c) < 500:
            return None
        best_rect = cv2.minAreaRect(c)

        return best_rect

    def get_contour_rect(self, mask):
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            return None
        c = max(contours, key = cv2.contourArea)
        if cv2.contourArea(c) < 500:
            return None
        return cv2.boundingRect(c)
        
    def isolate_largest_component(self, img):
        result = np.zeros((img.shape))
        labels, stats = cv2.connectedComponentsWithStats(img.astype(np.uint8), connectivity=8)[1:3]                   
        
        if len(stats) == 1:
            return img
        
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        result[labels == largest_label] = 255 

        return result