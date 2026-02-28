# gesture_model.py
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class GestureModel:
    def __init__(self, window_size_samples, sampling_rate):
        # İstediğiniz RandomForest parametreleri
        self.model = RandomForestClassifier(max_depth=None, min_samples_split=2, n_estimators=200)
        self.scaler = StandardScaler()
        self.window_size_samples = window_size_samples
        self.sampling_rate = sampling_rate
        self.gesture_labels = []
        
        # Calculate samples per window
        self.samples_per_window = window_size_samples 
        self.stride = self.samples_per_window // 2  # 50% overlap
    @staticmethod
    def extract_features(time_series_data):
        """
        Same feature extraction as before
        time_series_data shape: (time_steps, 34)
        Returns: 1D feature vector (170 features)
        """
        features = []
        for channel in range(time_series_data.shape[1]):
            channel_data = time_series_data[:, channel]
            features.extend([
                np.mean(channel_data),
                np.std(channel_data),
                np.min(channel_data),
                np.max(channel_data),
                np.max(channel_data) - np.min(channel_data),
            ])
        return np.array(features)
    
    
    def train(self, gesture_data_dict):
        """
        Train on segmented gesture data from calibration_data/pXnew
        gesture_data_dict: {gesture_name: [list of numpy arrays]}
        """
        print("Extracting features from gesture data with sliding windows...")
        X = []
        y = []
        
        self.gesture_labels = sorted(gesture_data_dict.keys())
        
        for gesture_name in self.gesture_labels:
            samples = gesture_data_dict[gesture_name]
            for time_series in samples:
                # Slide window with 50% overlap
                num_windows = (len(time_series) - self.samples_per_window) // self.stride + 1
                
                for i in range(num_windows):
                    start_idx = i * self.stride
                    end_idx = start_idx + self.samples_per_window
                    window = time_series[start_idx:end_idx]
                    
                    features = self.extract_features(window)
                    X.append(features)
                    y.append(gesture_name)
        
        X = np.array(X)
        y = np.array(y)

        # VERİYİ BÖLME İŞLEMİ BURADA YAPILIYOR:
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Sadece eğitim verisiyle ölçeklendirme yapıyoruz
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Eğitim
        self.model.fit(X_train_scaled, y_train)
        
        # Başarıyı test etme
        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Model Accuracy: %{accuracy * 100:.2f}")
        return accuracy

    
    def classify(self, features):
        """
        Classify a feature vector
        features: 1D numpy array (170 features)
        Returns: gesture name (string)
        """
        if len(features.shape) == 1:
            features = features.reshape(1, -1)
        
        features_scaled = self.scaler.transform(features)
        prediction = self.model.predict(features_scaled)
        return prediction[0]
    
    def save_model(self, filepath):
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'gesture_labels': self.gesture_labels,
            'window_size_samples': self.window_size_samples,  # Changed from window_size_ms
            'sampling_rate': self.sampling_rate,
            'samples_per_window': self.samples_per_window,
            'stride': self.stride
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"GestureModel saved to {filepath}")
    
    def load_model(self, filepath):
        if not os.path.exists(filepath):
            print(f"Model file {filepath} not found")
            return False
        
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.gesture_labels = model_data['gesture_labels']
        self.window_size_samples = model_data['window_size_samples'] 
        self.sampling_rate = model_data['sampling_rate']
        self.samples_per_window = model_data['samples_per_window']
        self.stride = model_data['stride']
        
        print(f"GestureModel loaded from {filepath}")
        return True