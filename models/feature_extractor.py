"""Feature extraction from trained model"""
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
