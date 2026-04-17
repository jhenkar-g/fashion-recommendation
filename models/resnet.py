"""ResNet architecture implementation"""
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
