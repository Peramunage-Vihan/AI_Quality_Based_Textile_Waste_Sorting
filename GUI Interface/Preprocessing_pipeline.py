import torch
import pickle
import joblib
import numpy as np
from sklearn.decomposition import PCA
from scipy.signal import savgol_filter


class PreprocessPipeline():
    def __init__(self,input_data, pca_model_path='pca_model.pkl', mean_spectrum_path='mean_spectram_of_traing_data.npy'):
        self.input_data = input_data
        self.snv_data = None
        self.savgol_data = None
        self.mean_centered_data = None
        self.pca_data = None
        self.pca_model = joblib.load(pca_model_path)
        self.training_mean_spectrum = np.load(mean_spectrum_path)

    def snv_normalization(self):
        """Applies Standard Normal Variate (SNV) to a single spectrum."""
        spectrum_array = np.array(self.input_data)
        mean_spectrum = np.mean(spectrum_array)
        std_spectrum = np.std(spectrum_array)
        
        if std_spectrum > 1e-8:
            snv_spectrum = (spectrum_array - mean_spectrum) / std_spectrum
        else:
            snv_spectrum = spectrum_array - mean_spectrum
            
        self.snv_data = snv_spectrum
        return self.snv_data
    
    def savgol_derivative(self, window_length=15, polyorder=2, deriv=2):
        """Applies a Savitzky-Golay filter to a single spectrum."""
        spectrum_array = np.array(self.snv_data)
        
        filtered_spectrum = savgol_filter(spectrum_array, 
                                        window_length=window_length, 
                                        polyorder=polyorder,                                        
                                        deriv=deriv)        
        self.savgol_data = filtered_spectrum
        return self.savgol_data

    def mean_centering(self):
        """Performs mean centering using the mean spectrum from the training data."""
        spectrum_array = np.array(self.savgol_data)
        
        # Subtract the training data's mean spectrum
        mean_centered_array = spectrum_array - self.training_mean_spectrum
        
        self.mean_centered_data = mean_centered_array
        return self.mean_centered_data

    def pca_transformation(self):
        """Transforms the data using the pre-fitted PCA model."""
        # Reshape data to be 2D array (1, n_features) for PCA transform
        data_to_transform = self.mean_centered_data.reshape(1, -1)
        
        # Transform the data using the loaded PCA model
        input_data_pca = self.pca_model.transform(data_to_transform)
        
        self.pca_data = input_data_pca.flatten() # Flatten to get 1D array
        return self.pca_data

    def run_pipeline(self,savgol_window_length=15, savgol_polyorder=2, savgol_deriv=2):
        """Runs the full preprocessing pipeline for a single spectrum."""
        self.snv_normalization()
        self.savgol_derivative(window_length=savgol_window_length,
                               polyorder=savgol_polyorder,
                               deriv=savgol_deriv)
        self.mean_centering()
        self.pca_transformation()
        return self.pca_data