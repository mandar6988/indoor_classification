

import numpy as np
from keras.models import load_model
from PIL import Image
import tensorflow as tf

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
input_shape = (224, 224)  
num_classes = 67  



class indoor_class:
    def __init__(self,filename):
        self.filename =filename
    
    def preprocess_image(self,image_path):
        img = load_img(image_path, target_size=input_shape)
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0  # Normalize the image pixels between 0 and 1
        return img_array
   


    def predictiondogcat(self):
        
        saved_model_path = "train_25.h5"
        model = load_model(saved_model_path)
        # model.summary()
        imagename = self.filename
        

        single_image = self.preprocess_image(imagename)
        prediction = model.predict(single_image)
        print(prediction)
        prediction_scores = prediction[0][0]
        


        predicted_class_index = np.argmax(prediction)
        prediction_scores = prediction[0][predicted_class_index]
        print(prediction_scores)
        class_mapping = {}
        with open("class_mapping.txt", "r") as file:
            for line in file:
                idx, label = line.strip().split()
                class_mapping[int(idx)] = label

        predicted_class_label = class_mapping[predicted_class_index]

        print(f"Predicted Class Label: {predicted_class_label}")
        print(prediction_scores)
        result={"prediction_scores":str(prediction_scores*100),
                "predicted_class_label":predicted_class_label
                }

        return result







       


