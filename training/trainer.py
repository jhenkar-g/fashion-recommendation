"""Training pipeline"""
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
