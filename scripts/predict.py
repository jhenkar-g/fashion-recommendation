"""Prediction script"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np, tensorflow as tf
from pathlib import Path
from config.config import config
from data.preprocessing import ImagePreprocessor
from models.resnet import create_fashion_resnet
from models.feature_extractor import FeatureExtractor, SimilarityMatcher
from inference.predictor import Predictor
from utils.logging import logger

def predict_single(image_path, checkpoint_path):
    logger.info(f"Predicting: {image_path}")
    model = create_fashion_resnet(num_classes=config.NUM_CLASSES)
    model.load_weights(checkpoint_path)
    preprocessor = ImagePreprocessor()
    predictor = Predictor(model, preprocessor, config)
    return predictor.predict_single(image_path)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', required=True)
    parser.add_argument('--checkpoint', required=True)
    args = parser.parse_args()
    result = predict_single(args.image, args.checkpoint)
    print(result)
