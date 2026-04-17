"""Training script"""
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
