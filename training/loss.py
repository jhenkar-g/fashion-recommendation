"""Loss functions"""
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
