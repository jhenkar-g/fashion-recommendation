#!/usr/bin/env python3
"""Complete automated setup for Fashion Recommendation System"""

import os
from pathlib import Path

# ALL FILES TO CREATE
FILES = {
    # CONFIG
    'config/__init__.py': '',
    'config/config.py': '''"""Configuration settings for fashion recommendation system"""

class Config:
    """Base configuration"""
    NUM_RESIDUAL_BLOCKS = 2
    NUM_CLASSES = 6
    IMG_HEIGHT = 64
    IMG_WIDTH = 64
    IMG_CHANNELS = 3
    BATCH_SIZE = 32
    VALIDATION_BATCH_SIZE = 25
    TEST_BATCH_SIZE = 25
    LEARNING_RATE = 0.01
    WEIGHT_DECAY = 0.00025
    FC_WEIGHT_DECAY = 0.00025
    NUM_EPOCHS = 100
    TRAIN_DATA_PATH = 'data/train_images.csv'
    VAL_DATA_PATH = 'data/val_images.csv'
    TEST_DATA_PATH = 'data/test_images.csv'
    CHECKPOINT_PATH = 'checkpoints/'
    LOG_PATH = 'logs/'
    ENABLE_LOCALIZATION = True
    FEATURE_DIM = 128
    USE_DATA_AUGMENTATION = True
    NORMALIZE = True
    IMAGENET_MEAN = [103.939, 116.799, 123.68]
    GLOBAL_STD = 68.76

class DevelopmentConfig(Config):
    DEBUG = True
    BATCH_SIZE = 16
    NUM_EPOCHS = 5

class ProductionConfig(Config):
    DEBUG = False
    BATCH_SIZE = 32
    NUM_EPOCHS = 100

config = ProductionConfig()
''',

    # DATA
    'data/__init__.py': '',
    'data/preprocessing.py': '''"""Data preprocessing utilities for fashion images"""
import cv2
import numpy as np
import pandas as pd
from pathlib import Path

class ImagePreprocessor:
    def __init__(self, img_height=64, img_width=64, imagenet_mean=None, global_std=68.76):
        self.img_height = img_height
        self.img_width = img_width
        self.imagenet_mean = imagenet_mean or [103.939, 116.799, 123.68]
        self.global_std = global_std
    
    def load_image(self, image_path):
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Failed to load image: {image_path}")
        return img
    
    def resize_image(self, img):
        return cv2.resize(img, (self.img_width, self.img_height))
    
    def normalize_image(self, img):
        img = img.astype(np.float32)
        img -= np.array(self.imagenet_mean)
        img /= self.global_std
        return img
    
    def preprocess(self, image_path, normalize=True):
        img = self.load_image(image_path)
        img = self.resize_image(img)
        if normalize:
            img = self.normalize_image(img)
        return img
    
    def batch_preprocess(self, image_paths, normalize=True):
        batch = []
        for path in image_paths:
            img = self.preprocess(path, normalize)
            batch.append(img)
        return np.array(batch)

class BoundingBoxNormalizer:
    @staticmethod
    def normalize_bbox(bbox, img_height, img_width):
        x1, y1, x2, y2 = bbox
        return np.array([x1/img_width, y1/img_height, x2/img_width, y2/img_height], dtype=np.float32)
    
    @staticmethod
    def denormalize_bbox(bbox, img_height, img_width):
        x1, y1, x2, y2 = bbox
        return np.array([x1*img_width, y1*img_height, x2*img_width, y2*img_height], dtype=np.int32)

def prepare_dataset_csv(image_dir, output_csv, category_mapping=None):
    image_dir = Path(image_dir)
    data = []
    if category_mapping is None:
        category_mapping = {folder.name: idx for idx, folder in enumerate(image_dir.iterdir()) if folder.is_dir()}
    for category, cat_id in category_mapping.items():
        category_dir = image_dir / category
        if not category_dir.exists():
            continue
        for img_path in category_dir.glob('*.jpg'):
            data.append({'image_path': str(img_path), 'category': cat_id, 'category_name': category})
    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)
    print(f"Dataset CSV created: {output_csv} ({len(df)} images)")
    return df
''',

    'data/data_loader.py': '''"""Data loading utilities"""
import pandas as pd
import numpy as np
from .preprocessing import ImagePreprocessor, BoundingBoxNormalizer

class FashionDataLoader:
    def __init__(self, csv_path, preprocessor=None, img_height=64, img_width=64):
        self.df = pd.read_csv(csv_path)
        self.preprocessor = preprocessor or ImagePreprocessor(img_height, img_width)
        self.img_height = img_height
        self.img_width = img_width
        self.bbox_normalizer = BoundingBoxNormalizer()
    
    def get_batch(self, indices, normalize=True, with_bbox=False):
        batch_data = self.df.iloc[indices]
        images, labels, bboxes = [], [], []
        for _, row in batch_data.iterrows():
            img = self.preprocessor.preprocess(row['image_path'], normalize=normalize)
            images.append(img)
            labels.append(row['category'])
            if with_bbox and 'x1' in row:
                bbox = np.array([row['x1'], row['y1'], row['x2'], row['y2']])
                bbox = self.bbox_normalizer.normalize_bbox(bbox, self.img_height, self.img_width)
                bboxes.append(bbox)
        images = np.array(images)
        labels = np.array(labels)
        if with_bbox:
            bboxes = np.array(bboxes)
            return images, labels, bboxes
        return images, labels
    
    def get_random_batch(self, batch_size, normalize=True, with_bbox=False):
        indices = np.random.choice(len(self.df), batch_size, replace=False)
        return self.get_batch(indices, normalize, with_bbox)
    
    def get_sequential_batches(self, batch_size, normalize=True, with_bbox=False):
        num_batches = len(self.df) // batch_size
        for i in range(num_batches):
            start_idx, end_idx = i * batch_size, (i+1) * batch_size
            indices = np.arange(start_idx, end_idx)
            yield self.get_batch(indices, normalize, with_bbox)
    
    def __len__(self):
        return len(self.df)
''',

    'data/augmentation.py': '''"""Data augmentation utilities"""
import cv2
import numpy as np

class ImageAugmentor:
    @staticmethod
    def random_flip(img, flip_prob=0.5):
        if np.random.random() < flip_prob:
            img = cv2.flip(img, 1)
        return img
    
    @staticmethod
    def random_rotation(img, angle_range=15):
        angle = np.random.uniform(-angle_range, angle_range)
        h, w = img.shape[:2]
        matrix = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
        img = cv2.warpAffine(img, matrix, (w, h))
        return img
    
    @staticmethod
    def random_brightness(img, brightness_range=0.2):
        factor = np.random.uniform(1 - brightness_range, 1 + brightness_range)
        img = cv2.convertScaleAbs(img, alpha=factor, beta=0)
        return np.clip(img, 0, 255).astype(np.uint8)
    
    @staticmethod
    def random_crop(img, crop_size=56):
        h, w = img.shape[:2]
        y, x = np.random.randint(0, h - crop_size), np.random.randint(0, w - crop_size)
        return img[y:y+crop_size, x:x+crop_size]
    
    @staticmethod
    def augment(img, use_flip=True, use_rotation=False, use_brightness=False, use_crop=False):
        if use_flip:
            img = ImageAugmentor.random_flip(img)
        if use_rotation:
            img = ImageAugmentor.random_rotation(img)
        if use_brightness:
            img = ImageAugmentor.random_brightness(img)
        if use_crop:
            img = ImageAugmentor.random_crop(img)
        return img
''',

    # MODELS
    'models/__init__.py': '',
    'models/resnet.py': '''"""ResNet architecture implementation"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class ResidualBlock(layers.Layer):
    def __init__(self, filters, stride=1, **kwargs):
        super(ResidualBlock, self).__init__(**kwargs)
        self.filters = filters
        self.stride = stride
        self.conv1 = layers.Conv2D(filters, 3, strides=stride, padding='same')
        self.bn1 = layers.BatchNormalization()
        self.relu1 = layers.ReLU()
        self.conv2 = layers.Conv2D(filters, 3, strides=1, padding='same')
        self.bn2 = layers.BatchNormalization()
        if stride != 1:
            self.conv_skip = layers.Conv2D(filters, 1, strides=stride)
            self.bn_skip = layers.BatchNormalization()
        else:
            self.conv_skip = None
        self.relu_out = layers.ReLU()
    
    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        skip = self.conv_skip(inputs) if self.conv_skip else inputs
        if self.conv_skip:
            skip = self.bn_skip(skip)
        x = x + skip
        x = self.relu_out(x)
        return x

class FashionResNet(keras.Model):
    def __init__(self, num_classes=6, num_blocks=2, **kwargs):
        super(FashionResNet, self).__init__(**kwargs)
        self.input_conv = layers.Conv2D(64, 7, strides=2, padding='same')
        self.input_bn = layers.BatchNormalization()
        self.input_relu = layers.ReLU()
        self.input_pool = layers.MaxPooling2D(3, strides=2, padding='same')
        self.res_blocks = [ResidualBlock(64 * (2 ** i), stride=2 if i > 0 else 1) for i in range(num_blocks)]
        self.gap = layers.GlobalAveragePooling2D()
        self.fc1 = layers.Dense(128, activation='relu')
        self.dropout1 = layers.Dropout(0.5)
        self.fc_class = layers.Dense(num_classes)
        self.fc2 = layers.Dense(128, activation='relu')
        self.dropout2 = layers.Dropout(0.5)
        self.fc_bbox = layers.Dense(4)
    
    def call(self, inputs, training=False):
        x = self.input_conv(inputs)
        x = self.input_bn(x, training=training)
        x = self.input_relu(x)
        x = self.input_pool(x)
        for block in self.res_blocks:
            x = block(x, training=training)
        x = self.gap(x)
        features = x
        x_class = self.fc1(x)
        x_class = self.dropout1(x_class, training=training)
        logits = self.fc_class(x_class)
        x_bbox = self.fc2(x)
        x_bbox = self.dropout2(x_bbox, training=training)
        bboxes = self.fc_bbox(x_bbox)
        return logits, bboxes, features

def create_fashion_resnet(num_classes=6, num_blocks=2):
    return FashionResNet(num_classes=num_classes, num_blocks=num_blocks)
''',

    'models/feature_extractor.py': '''"""Feature extraction from trained model"""
import numpy as np
import tensorflow as tf
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

class FeatureExtractor:
    def __init__(self, model):
        self.model = model
    
    def extract(self, image_batch):
        _, _, features = self.model(image_batch, training=False)
        return features.numpy()
    
    def extract_single(self, image):
        image = np.expand_dims(image, axis=0)
        return self.extract(image)[0]

class SimilarityMatcher:
    def __init__(self, feature_extractor, similarity_metric='cosine'):
        self.feature_extractor = feature_extractor
        self.similarity_metric = similarity_metric
        self.feature_database = None
        self.image_paths = []
    
    def build_database(self, image_batch, image_paths):
        self.feature_database = self.feature_extractor.extract(image_batch)
        self.image_paths = image_paths
    
    def find_matches(self, query_image, top_k=5):
        if self.feature_database is None:
            raise ValueError("Feature database not built")
        query_feature = self.feature_extractor.extract_single(query_image).reshape(1, -1)
        similarities = cosine_similarity(query_feature, self.feature_database)[0] if self.similarity_metric == 'cosine' else -euclidean_distances(query_feature, self.feature_database)[0]
        top_indices = np.argsort(similarities)[::-1][:top_k]
        results = [(self.image_paths[idx], float(similarities[idx])) for idx in top_indices]
        return results
''',

    # TRAINING
    'training/__init__.py': '',
    'training/loss.py': '''"""Loss functions"""
import tensorflow as tf

class MultiTaskLoss(tf.keras.losses.Loss):
    def __init__(self, alpha=1.0, beta=1.0, **kwargs):
        super(MultiTaskLoss, self).__init__(**kwargs)
        self.alpha = alpha
        self.beta = beta
        self.ce_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.mse_loss = tf.keras.losses.MeanSquaredError()
    
    def call(self, y_true, y_pred):
        class_labels, bbox_labels = y_true
        class_logits, bbox_pred = y_pred
        ce = self.ce_loss(class_labels, class_logits)
        mse = self.mse_loss(bbox_labels, bbox_pred)
        return self.alpha * ce + self.beta * mse

def get_loss_function(task='multi_task', alpha=1.0, beta=1.0):
    if task == 'multi_task':
        return MultiTaskLoss(alpha=alpha, beta=beta)
    elif task == 'classification':
        return tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    else:
        return tf.keras.losses.MeanSquaredError()
''',

    'training/metrics.py': '''"""Evaluation metrics"""
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def compute_metrics(y_true, y_pred, num_classes=None):
    y_pred_labels = np.argmax(y_pred, axis=1)
    return {
        'accuracy': accuracy_score(y_true, y_pred_labels),
        'precision': precision_score(y_true, y_pred_labels, average='weighted', zero_division=0),
        'recall': recall_score(y_true, y_pred_labels, average='weighted', zero_division=0),
        'f1': f1_score(y_true, y_pred_labels, average='weighted', zero_division=0)
    }
''',

    'training/trainer.py': '''"""Training pipeline"""
import tensorflow as tf
import numpy as np
from pathlib import Path
import json

class Trainer:
    def __init__(self, model, loss_fn, optimizer, train_loader, val_loader, config):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.checkpoint_dir = Path(config.CHECKPOINT_PATH)
        self.checkpoint_dir.mkdir(exist_ok=True)
    
    @tf.function
    def train_step(self, images, labels, bboxes):
        with tf.GradientTape() as tape:
            class_logits, bbox_pred, _ = self.model(images, training=True)
            loss = self.loss_fn((labels, bboxes), (class_logits, bbox_pred))
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return loss
    
    @tf.function
    def val_step(self, images, labels, bboxes):
        class_logits, bbox_pred, _ = self.model(images, training=False)
        return self.loss_fn((labels, bboxes), (class_logits, bbox_pred))
    
    def train_epoch(self, epoch):
        total_loss = 0
        num_batches = 0
        for images, labels, bboxes in self.train_loader.get_sequential_batches(self.config.BATCH_SIZE, with_bbox=True):
            loss = self.train_step(tf.convert_to_tensor(images), tf.convert_to_tensor(labels), tf.convert_to_tensor(bboxes))
            total_loss += loss.numpy()
            num_batches += 1
        avg_loss = total_loss / num_batches
        self.train_losses.append(float(avg_loss))
        return avg_loss
    
    def validate(self):
        total_loss = 0
        num_batches = 0
        for images, labels, bboxes in self.val_loader.get_sequential_batches(self.config.VALIDATION_BATCH_SIZE, with_bbox=True):
            loss = self.val_step(tf.convert_to_tensor(images), tf.convert_to_tensor(labels), tf.convert_to_tensor(bboxes))
            total_loss += loss.numpy()
            num_batches += 1
        avg_loss = total_loss / num_batches
        self.val_losses.append(float(avg_loss))
        return avg_loss
    
    def train(self, num_epochs):
        for epoch in range(num_epochs):
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate()
            print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint("best_model.ckpt")
        self.save_checkpoint("final_model.ckpt")
        self.save_history()
    
    def save_checkpoint(self, filename):
        filepath = self.checkpoint_dir / filename
        self.model.save_weights(str(filepath))
        print(f"Checkpoint saved: {filepath}")
    
    def load_checkpoint(self, filename):
        filepath = self.checkpoint_dir / filename
        self.model.load_weights(str(filepath))
        print(f"Checkpoint loaded: {filepath}")
    
    def save_history(self):
        history = {'train_losses': self.train_losses, 'val_losses': self.val_losses}
        filepath = self.checkpoint_dir / 'history.json'
        with open(filepath, 'w') as f:
            json.dump(history, f)
''',

    # INFERENCE
    'inference/__init__.py': '',
    'inference/predictor.py': '''"""Inference pipeline"""
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
''',

    'inference/matcher.py': '''"""Clothing matching"""
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class ClothingMatcher:
    def __init__(self, feature_extractor):
        self.feature_extractor = feature_extractor
        self.wardrobe_features = None
        self.wardrobe_metadata = []
    
    def add_to_wardrobe(self, images, metadata):
        features = self.feature_extractor.extract(images)
        self.wardrobe_features = features if self.wardrobe_features is None else np.vstack([self.wardrobe_features, features])
        self.wardrobe_metadata.extend(metadata)
    
    def find_matching_pairs(self, query_image, top_k=5, similarity_threshold=0.7):
        query_feature = self.feature_extractor.extract_single(query_image).reshape(1, -1)
        similarities = cosine_similarity(query_feature, self.wardrobe_features)[0]
        valid_indices = np.where(similarities >= similarity_threshold)[0]
        top_indices = np.argsort(similarities[valid_indices])[::-1][:top_k]
        matches = [{'metadata': self.wardrobe_metadata[valid_indices[idx]], 'similarity_score': float(similarities[valid_indices[idx]])} for idx in top_indices]
        return matches
''',

    # UTILS
    'utils/__init__.py': '',
    'utils/helpers.py': '''"""Helper utilities"""
import os, json
from pathlib import Path

def ensure_directory(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def save_json(data, filepath):
    ensure_directory(os.path.dirname(filepath))
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)

def load_json(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def get_image_files(directory, extensions=('.jpg', '.jpeg', '.png')):
    image_files = []
    for ext in extensions:
        image_files.extend(Path(directory).glob(f'**/*{ext}'))
    return sorted(image_files)
''',

    'utils/logging.py': '''"""Logging configuration"""
import logging
from pathlib import Path

def setup_logging(log_dir='logs', log_file='training.log'):
    Path(log_dir).mkdir(exist_ok=True)
    log_path = Path(log_dir) / log_file
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', handlers=[logging.FileHandler(log_path), logging.StreamHandler()])
    return logging.getLogger(__name__)

logger = setup_logging()
''',

    # SCRIPTS
    'scripts/__init__.py': '',
    'scripts/train.py': '''"""Training script"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensorflow as tf
from config.config import config
from data.data_loader import FashionDataLoader
from data.preprocessing import ImagePreprocessor
from models.resnet import create_fashion_resnet
from training.trainer import Trainer
from training.loss import get_loss_function
from utils.logging import logger

def train():
    logger.info("Starting training...")
    preprocessor = ImagePreprocessor(img_height=config.IMG_HEIGHT, img_width=config.IMG_WIDTH)
    train_loader = FashionDataLoader(config.TRAIN_DATA_PATH, preprocessor=preprocessor)
    val_loader = FashionDataLoader(config.VAL_DATA_PATH, preprocessor=preprocessor)
    logger.info(f"Train: {len(train_loader)}, Val: {len(val_loader)}")
    model = create_fashion_resnet(num_classes=config.NUM_CLASSES, num_blocks=config.NUM_RESIDUAL_BLOCKS)
    loss_fn = get_loss_function('multi_task')
    optimizer = tf.keras.optimizers.SGD(learning_rate=config.LEARNING_RATE)
    trainer = Trainer(model, loss_fn, optimizer, train_loader, val_loader, config)
    trainer.train(num_epochs=config.NUM_EPOCHS)
    logger.info("Training complete!")

if __name__ == '__main__':
    train()
''',

    'scripts/evaluate.py': '''"""Evaluation script"""
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
''',

    'scripts/predict.py': '''"""Prediction script"""
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
''',

    # ROOT
    'requirements.txt': '''tensorflow>=2.10.0
numpy>=1.20.0
pandas>=1.3.0
opencv-python>=4.5.0
scikit-learn>=1.0.0
Pillow>=8.0.0
matplotlib>=3.4.0''',

    'setup.py': '''from setuptools import setup, find_packages
setup(name='fashion-recommendation', version='0.1.0', description='Fashion recommendation system', packages=find_packages(), python_requires='>=3.7', install_requires=['tensorflow>=2.10.0', 'numpy>=1.20.0', 'pandas>=1.3.0', 'opencv-python>=4.5.0', 'scikit-learn>=1.0.0', 'Pillow>=8.0.0', 'matplotlib>=3.4.0'])
''',

    '.gitignore': '''__pycache__/
*.py[cod]
*.egg-info/
build/
dist/
venv/
data/
checkpoints/
logs/
*.ckpt
.DS_Store
'''
}

def setup():
    print("🚀 Creating Fashion Recommendation Project...\n")
    
    for filepath, content in FILES.items():
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            f.write(content)
        print(f"✅ {filepath}")
    
    print(f"\n✅ Created {len(FILES)} files!")
    print("\n📋 Next Steps:")
    print("1. pip install -r requirements.txt")
    print("2. Prepare data in data/ folder")
    print("3. python scripts/train.py")

if __name__ == '__main__':
    setup()