"""Evaluation script"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np, tensorflow as tf
from config.config import config
from data.data_loader import FashionDataLoader
from data.preprocessing import ImagePreprocessor
from models.resnet import create_fashion_resnet
from training.metrics import compute_metrics
from utils.logging import logger

def evaluate(checkpoint_path):
    logger.info(f"Evaluating: {checkpoint_path}")
    model = create_fashion_resnet(num_classes=config.NUM_CLASSES, num_blocks=config.NUM_RESIDUAL_BLOCKS)
    model.load_weights(checkpoint_path)
    preprocessor = ImagePreprocessor(img_height=config.IMG_HEIGHT, img_width=config.IMG_WIDTH)
    test_loader = FashionDataLoader(config.TEST_DATA_PATH, preprocessor=preprocessor)
    predictions, labels = [], []
    for images, test_labels in test_loader.get_sequential_batches(config.TEST_BATCH_SIZE):
        class_logits, _, _ = model(tf.convert_to_tensor(images), training=False)
        predictions.extend(class_logits.numpy())
        labels.extend(test_labels)
    metrics = compute_metrics(np.array(labels), np.array(predictions))
    logger.info(f"Metrics: {metrics}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    evaluate(parser.parse_args().checkpoint)
