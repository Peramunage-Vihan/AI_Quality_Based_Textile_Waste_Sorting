import numpy as np

def defect_counter(results, defect_count=None):
    if defect_count is None:
        defect_count = np.zeros(5)
    else:
        defect_count.fill(0)  # Reset the counter
    
    # Handle the case where results might be a list or single result
    if hasattr(results, 'boxes') and results.boxes is not None:
        for box in results.boxes:
            class_id = int(box.cls[0])
            if class_id < 5:  # Ensure class_id is within bounds
                defect_count[class_id] += 1
    
    return defect_count
