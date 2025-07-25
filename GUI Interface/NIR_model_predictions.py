import torch
import numpy as np
from Preprocessing_pipeline import PreprocessPipeline
from NIR_Architecture import SpectralCNN1D, NIR_model

def predict_nir_spectrum(raw_spectrum):
    pipeline = PreprocessPipeline(input_data = raw_spectrum)

    pca_data = pipeline.run_pipeline()

    input_tensor = torch.tensor(pca_data, dtype=torch.float32)
 
    input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        predictions = NIR_model(input_tensor)

    return predictions





