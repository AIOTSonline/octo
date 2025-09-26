import os
from ultralytics import YOLO

# --- Configuration ---
# Path to the pre-trained PyTorch model you want to convert.
MODEL_PATH = 'runs/detect/train6/weights/best.pt'


def convert_to_int8_tflite_simplified(model_path=MODEL_PATH):
    """
    Loads a YOLOv8 PyTorch model and exports it to an 8-bit integer
    quantized TensorFlow Lite (.tflite) format.
    This version simplifies the calibration by letting Ultralytics handle it internally.
    """
    print("--- Starting YOLOv8 to INT8 TFLite Conversion (Simplified Calibration) ---")

    # --- Step 1: Validate that the model file exists ---
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at '{model_path}'")
        print("Please ensure the path is correct and you are running the script from the 'Arishna Internship' directory.")
        return

    print(f"Using model: {model_path}")

    try:
        # --- Step 2: Load the YOLOv8 model ---
        print("\nLoading the YOLOv8 model...")
        model = YOLO(model_path)
        print("Model loaded successfully.")

        # --- Step 3: Export the model with INT8 quantization ---
        print("\nExporting model to INT8 TFLite format...")
        print("Ultralytics will attempt to perform internal calibration.")

        # To create an INT8 model, set 'int8=True'.
        # The 'data' argument is removed to let ultralytics manage calibration.
        model.export(format='tflite', int8=True)

        print("\n--- Conversion Complete! ---")
        print(f"INT8 quantized TFLite model saved successfully.")
        print(f"Output file can be found in the same directory as the original model: {
              os.path.dirname(model_path)}")

    except Exception as e:
        print(f"\nAn error occurred during the conversion process: {e}")
        print("Please ensure all dependencies are correctly installed and that your system has sufficient resources.")


if __name__ == '__main__':
    convert_to_int8_tflite_simplified(MODEL_PATH)
