from sklearn import svm
from sklearn.preprocessing import StandardScaler
import numpy as np
import pickle
import os

class RestDetector:
    #  binary SVM for understanding if data is rest or not
    def __init__(self, window_size):  # samples, not ms
        self.model = svm.SVC(kernel='rbf', C=1.0, gamma='scale')
        self.scaler = StandardScaler()
        self.window_size = window_size

    @staticmethod
    def extract_features(time_series_data):
        """
        IDENTICAL to gesture_model.extract_features()
        time_series_data shape: (time_steps, 34)
        Returns: 1D feature vector (170 features = 34 channels × 5 features)
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
        
    def train(self):
        """
        Train binary SVM on ALL participant data
        - rest class: calibration_data/p{1-6}rest/*.npy
        - non-rest class: processed_data/p{1-6}/*.npy
        Uses sliding windows with 50% overlap (stride = window_size // 2)
        """
        import glob
        
        print("Loading rest data from calibration_data/p{1-6}rest...")
        rest_samples = []
        for participant_id in range(1, 7):
            folder = f'rows_deleted/p{participant_id}'
            if os.path.exists(folder):
                files = glob.glob(f'{folder}/*.npy')
                # FILTER: only files starting with "rest"
                files = [f for f in files if os.path.basename(f).startswith('rest')]
                for file in files:
                    data = np.load(file)
                    rest_samples.append(data)
                    print(f"  ✓ Loaded: {os.path.basename(file)} (shape: {data.shape})")
        
        print(f"\nLoading non-rest data from rows_deleted/p{{1-6}}...")
        not_rest_samples = []
        for participant_id in range(1, 7):
            folder = f'rows_deleted/p{participant_id}'
            if os.path.exists(folder):
                files = glob.glob(f'{folder}/*.npy') 
                # FILTER: only files NOT starting with "rest"
                files = [f for f in files if not os.path.basename(f).startswith('rest')]
                for file in files:
                    data = np.load(file)
                    not_rest_samples.append(data)
                    print(f"  ✓ Loaded: {os.path.basename(file)} (shape: {data.shape})")
        
        if len(rest_samples) == 0 or len(not_rest_samples) == 0:
            print("ERROR: Need both rest and non-rest data!")
            return False
        
        print(f"\nExtracting features with sliding windows (window_size={self.window_size}, stride={self.window_size//2})...")
        
        X = []
        y = []
        stride = self.window_size // 2  # 50% overlap
        
        # Process rest samples (label = 0)
        for time_series in rest_samples:
            num_windows = (len(time_series) - self.window_size) // stride + 1
            for i in range(num_windows):
                start_idx = i * stride
                end_idx = start_idx + self.window_size
                window = time_series[start_idx:end_idx]
                
                features = self.extract_features(window)
                X.append(features)
                y.append(0)  # rest class
        
        # Process non-rest samples (label = 1)
        for time_series in not_rest_samples:
            num_windows = (len(time_series) - self.window_size) // stride + 1
            for i in range(num_windows):
                start_idx = i * stride
                end_idx = start_idx + self.window_size
                window = time_series[start_idx:end_idx]
                
                features = self.extract_features(window)
                X.append(features)
                y.append(1)  # non-rest class
        
        X = np.array(X)
        y = np.array(y)
        
        print(f"\nTraining binary SVM on {len(X)} windows:")
        print(f"  - Rest samples: {np.sum(y == 0)}")
        print(f"  - Non-rest samples: {np.sum(y == 1)}")
        
        # Normalize features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train SVM
        self.model.fit(X_scaled, y)
        
        print("✓ RestDetector training complete!")
        return True
    def predict(self, window_data):
        """
        Predict if window is rest or not-rest
        Args:
            window_data: numpy array of shape (window_size, 34)
        Returns:
            True if rest, False if not-rest
        """
        # Extract features
        features = self.extract_features(window_data)
        
        # Reshape to 2D for scaler
        if len(features.shape) == 1:
            features = features.reshape(1, -1)
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Predict (0 = rest, 1 = non-rest)
        prediction = self.model.predict(features_scaled)
        
        # Return True if rest (0), False if non-rest (1)
        return prediction[0] == 0
    def save_model(self, filepath):
        """Save model, scaler, and window_size to file"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'window_size': self.window_size
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"RestDetector saved to {filepath}")

    def load_model(self, filepath):
        """Load model, scaler, and window_size from file"""
        if not os.path.exists(filepath):
            print(f"Model file {filepath} not found")
            return False
        
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.window_size = model_data['window_size']
        
        print(f"RestDetector loaded from {filepath}")
        return True