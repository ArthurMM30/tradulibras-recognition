#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import keras


class KeyPointClassifier(object):
    def __init__(
        self,
        model_path='model/keypoint_classifier/keypoint_classifier.keras',
    ):
        self.model = keras.models.load_model(model_path)

    def __call__(
        self,
        landmark_list,
    ):
        prediction = self.model.predict(np.array([landmark_list], dtype=np.float32))
        
        predict_result = np.squeeze(prediction)
        
        return predict_result