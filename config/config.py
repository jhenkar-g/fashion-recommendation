"""Configuration settings for fashion recommendation system"""

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
