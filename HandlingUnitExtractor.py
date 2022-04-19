import numpy as np
import segmentation_models as sm
import cv2
from PIL import Image
sm.set_framework('tf.keras')

class HandlingUnitExtractor():
    def __init__(self):
        model_file = 'best_model_handling_unit.h5'
        self.model_img_size = (512, 512)
        self.model = sm.Unet('efficientnetb3', classes=1, activation='sigmoid')
        self.model.load_weights(model_file) 
        self.preprocessor = sm.get_preprocessing('efficientnetb3')
        
    def extract(self, img):
        h, w = img.shape[:2]
        image = cv2.resize(img, self.model_img_size, interpolation = cv2.INTER_AREA)
        image = self.preprocessor(image)
        
        mask = self.model.predict(np.expand_dims(image, axis=0)).round().squeeze()
        mask = cv2.resize(mask, (w, h), interpolation = cv2.INTER_AREA)
        
        mask = self.isolate_largest_component(mask)
        
        return mask.astype(np.uint8)

    def isolate_largest_component(self, img):
        result = np.zeros((img.shape))
        labels, stats = cv2.connectedComponentsWithStats(img.astype(np.uint8), connectivity=8)[1:3]                   
        
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        result[labels == largest_label] = 255 

        return result