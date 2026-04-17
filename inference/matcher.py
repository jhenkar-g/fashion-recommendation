"""Clothing matching"""
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
