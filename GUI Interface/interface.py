import gradio as gr
import torch
import torch.nn as nn
from torchvision import models, transforms
from ultralytics import YOLO
from PIL import Image
import numpy as np
import random
import cv2
from result_extract_yolo import defect_counter
from Fuzzy_Inferrence_System import calculate_sorting
from NIR_model_predictions import predict_nir_spectrum


DEFECT_COUNTER = np.zeros(5)  # Assuming 5 classes for defects
COLOR_CLASS_NAMES = ['blend', 'dark_shade', 'light_shade', 'white']
COLOR_CLASS_COUNTER = np.zeros(len(COLOR_CLASS_NAMES))
PRE_DEFECT_COUNT = 4000
NUM_CLASSES = len(COLOR_CLASS_NAMES)
DEVICE = torch.device("cuda"if torch.cuda.is_available() else "cpu")
MODEL_WEIGHT_PATH_COLOR = "best_Efficientnet_model_v2.pth"
MODEL_WEIGHT_PATH_DEFECT = "best.pt"
TOTAL_DMG_PERCENTAGE = 0.0
DAMAGE_PERCENTAGES = np.zeros(5)

#color Detection Model
color_detection = models.efficientnet_v2_s()

num_ftrs = color_detection.classifier[-1].in_features
color_detection.classifier[-1]  = nn.Sequential(
    nn.Linear(num_ftrs, 256),
    nn.Hardswish(),
    nn.Dropout(p=0.5,inplace=True),
    nn.Linear(256, NUM_CLASSES),
)

color_detection.load_state_dict(torch.load(MODEL_WEIGHT_PATH_COLOR,map_location=DEVICE))
color_detection.to(DEVICE)
color_detection.eval()

preprocess_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
# defect detetion model
defect_detection = YOLO(MODEL_WEIGHT_PATH_DEFECT)



print("✅ Models loaded successfully. Starting Gradio interface...")

def random_spectrum_generator():
    random_spectrum = [round(random.uniform(0.0, 1.0), 6) for _ in range(4149)]
    spectrum_str = ",".join(map(str, random_spectrum))

    return spectrum_str

def handle_nir_prediction(spectrum_str):
    """
    Parses the spectrum string, runs the NIR model, and returns the fiber composition.
    """
    if not spectrum_str:
        return "Please provide NIR spectrum data as comma-separated values."
    try:
        # Convert comma-separated string to a list of floats
        raw_spectrum = [float(x.strip()) for x in spectrum_str.split(',')]
        
        # Predict using the NIR model
        predictions = predict_nir_spectrum(raw_spectrum)
        
        # Process predictions to get fiber composition
        predictions[predictions < 0] = 0
        fiber_comp = round(predictions.max().item() * 100, 1)
        if (predictions.argmax().item() == 0):
            fiber_comp = 100- fiber_comp
        else:
            fiber_comp = fiber_comp
    
        return f"{fiber_comp}%"
    except ValueError:
        return "Error: Invalid input. Please ensure data is comma-separated numbers."
    except Exception as e:
        return f"An error occurred: {e}"

def predict_on_frame(frame):
    global DEFECT_COUNTER
    global DAMAGE_PERCENTAGES
    global TOTAL_DMG_PERCENTAGE
    """
    This function takes a webcam frame, runs both models,
    and returns the frame with annotations drawn on it.
    """
    if frame is None:
        return None, 0, 0, 0, 0, 0

    # --- Defect Detection (YOLOv8) ---
    defect_results = defect_detection(frame, verbose=False)[0]

    DEFECT_COUNTER += defect_counter(defect_results)

    total_dmg = DEFECT_COUNTER.sum()
    # Handle division by zero
    if total_dmg > 0:
        DAMAGE_PERCENTAGES = DEFECT_COUNTER/PRE_DEFECT_COUNT
        for i in range(len(DAMAGE_PERCENTAGES)):
            if DEFECT_COUNTER[i] > PRE_DEFECT_COUNT:
                DAMAGE_PERCENTAGES[i] = 1
    else:
        DAMAGE_PERCENTAGES = np.zeros(5)

    # Draw bounding boxes from YOLO results onto the frame
    annotated_frame = defect_results.plot()

    # --- Color Classification (EfficientNetv2) ---
    pil_image = Image.fromarray(frame)
    input_tensor = preprocess_transform(pil_image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        color_output = color_detection(input_tensor)
        _, predicted_idx = torch.max(color_output, 1)
        predicted_color = COLOR_CLASS_NAMES[predicted_idx.item()]
        COLOR_CLASS_COUNTER[predicted_idx.item()] += 1
    
    # Add the color prediction text to the frame
    color_label = f"Overall Color: {predicted_color}"
    cv2.putText(
        annotated_frame, 
        color_label, 
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 
        1,
        (0, 255, 0),
        2
    )

    # Convert percentages to values between 0 and 1 for progress bars
    return (annotated_frame, 
            float(DAMAGE_PERCENTAGES[0]), 
            float(DAMAGE_PERCENTAGES[1]), 
            float(DAMAGE_PERCENTAGES[2]), 
            float(DAMAGE_PERCENTAGES[3]), 
            float(DAMAGE_PERCENTAGES[4]))


def get_sorting_results():
    """
    Calls the fuzzy logic system with the currently stored data.
    """
    # Ensure DAMAGE_PERCENTAGES exists and color has been detected
    if not np.any(DAMAGE_PERCENTAGES) or not np.any(COLOR_CLASS_COUNTER):
        return "Please run the webcam feed first to collect data.", "", ""

    reusability, recyclability, downgrade = calculate_sorting(
        DAMAGE_PERCENTAGES, 
        COLOR_CLASS_COUNTER
    )
    
    reusability_txt = f"{reusability:.2f}"
    recyclability_txt = f"{recyclability:.2f}"
    downgrade_txt = f"{downgrade:.2f}"

    return reusability_txt, recyclability_txt, downgrade_txt


def list_available_cameras():
    """List all available cameras"""
    available_cameras = []
    for i in range(10):  # Check first 10 camera indices
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available_cameras.append(i)
            cap.release()
    return available_cameras

print("Available cameras:", list_available_cameras())

# --- 3. CREATE AND LAUNCH THE GRADIO INTERFACE ---

class_names = defect_detection.names if hasattr(defect_detection, 'names') else {0: "Class_0", 1: "Class_1", 2: "Class_2", 3: "Class_3", 4: "Class_4"}

with gr.Blocks() as iface:
    gr.Markdown(
        """
        # 🔬 Real-Time Defect and Color Detection
        This interface uses a YOLOv8 model to detect defects and a EfficientNetv2 model to classify color. 
        1. Run the webcam stream to analyze the item.
        2. Click 'Calculate Sorting Results' to run the fuzzy logic analysis.
        """
    )
    with gr.Row():
        with gr.Column():
            webcam_input = gr.Image(sources=["webcam"], streaming=True)
            defect_sliders = [
                gr.Slider(minimum=0, maximum=1, label=f"Defect: {class_names.get(i, f'Class_{i}')}", interactive=False) for i in range(5)
            ]
        with gr.Column():
            annotated_output = gr.Image(label="Annotated Frame")
            calculate_btn = gr.Button("Calculate Sorting Results")
            with gr.Group():
                reusability_out = gr.Textbox(label="Reusability Score")
                recyclability_out = gr.Textbox(label="Recyclability Score")
                downgrade_out = gr.Textbox(label="Downgrade Score")
    
    gr.Markdown("---")
    gr.Markdown("## 🔬 NIR Spectrum Analysis")
    with gr.Row():
        with gr.Column(scale=3):
            nir_input = gr.Textbox(
                label="NIR Spectrum Data",
                placeholder="Paste comma-separated spectrum values here...",
                lines=5
            )
            generate_btn = gr.Button("Generate Random Spectrum")
        with gr.Column(scale=1):
            nir_predict_btn = gr.Button("Predict Fiber Composition")
            fiber_comp_out = gr.Textbox(label="Fiber Composition", interactive=False)


    # Connect webcam processing
    webcam_input.stream(
        fn=predict_on_frame,
        inputs=webcam_input,
        outputs=[annotated_output] + defect_sliders
    )

    # Connect button to fuzzy logic function
    calculate_btn.click(
        fn=get_sorting_results,
        inputs=None,
        outputs=[reusability_out, recyclability_out, downgrade_out]
    )

    generate_btn.click(
        fn=random_spectrum_generator,
        inputs=None,
        outputs=nir_input
    )
    # Connect NIR prediction
    nir_predict_btn.click(
        fn=handle_nir_prediction,
        inputs=nir_input,
        outputs=fiber_comp_out
    )




if __name__ == "__main__":
    iface.launch()