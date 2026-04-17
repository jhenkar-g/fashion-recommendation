"""Inference pipeline"""
import numpy as np
import tensorflow as tf

class Predictor:
    def __init__(self, model, preprocessor, config):
        self.model = model
        self.preprocessor = preprocessor
        self.config = config
    
    def predict_single(self, image_path):
        image = self.preprocessor.preprocess(image_path)
        image = np.expand_dims(image, axis=0)
        class_logits, bbox_pred, features = self.model(tf.convert_to_tensor(image), training=False)
        class_prob = tf.nn.softmax(class_logits).numpy()[0]
        predicted_class = np.argmax(class_prob)
        return {'predicted_class': int(predicted_class), 'confidence': float(class_prob[predicted_class]), 'class_probabilities': class_prob.tolist(), 'bbox': bbox_pred.numpy()[0].tolist(), 'features': features.numpy()[0].tolist()}
    
    def predict_batch(self, image_paths):
        results = []
        for path in image_paths:
            result = self.predict_single(path)
            result['image_path'] = str(path)
            results.append(result)
        return results
