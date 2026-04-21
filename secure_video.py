import cv2
import time
import onnxruntime as ort
from tiling import detect_tiled
import os
import time

# ──────────────────────────────────────────────────────────────────────
# 1. Configuration (Matching nudeblock.py)
# ──────────────────────────────────────────────────────────────────────
MODEL_PATH = "nudenet_v8.onnx"
CONF_THRESHOLD = 0.1          
IOU_THRESHOLD = 0.45           
INFER_INPUT_SIZE = 640         
CLASSES_TO_MASK = set(range(18)) # Using a set for faster lookups, masking all 18 classes

INPUT_VIDEO = 'hot.mp4'#str(input("Enter the input video file name: "))  # Replace with your input video name
OUTPUT_VIDEO = "output.mp4"    # The censored output video

def main():
    # ──────────────────────────────────────────────────────────────────────
    # 2. Initialize Model (GPU if available, else CPU)
    # ──────────────────────────────────────────────────────────────────────
    providers = ort.get_available_providers()
    if "CUDAExecutionProvider" in providers:
        chosen = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        print("[VideoBlock] Using CUDA GPU for inference.")
    else:
        chosen = ["CPUExecutionProvider"]
        print("[VideoBlock] CUDA not available – falling back to CPU.")
        
    session = ort.InferenceSession(MODEL_PATH, providers=chosen)
    input_name = session.get_inputs()[0].name

    # ──────────────────────────────────────────────────────────────────────
    # 3. Open Video and Setup Writer
    # ──────────────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(INPUT_VIDEO)
    if not cap.isOpened():
        print(f"[Error] Could not open video file: {INPUT_VIDEO}")
        return

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Setup the OpenCV Video Writer
    # mp4v is the standard codec for mp4 containers in OpenCV
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

    print(f"[VideoBlock] Processing {width}x{height} video at {fps} FPS.")
    print(f"[VideoBlock] Total frames to process: {total_frames}")

    # ──────────────────────────────────────────────────────────────────────
    # 4. Process Frames
    # ──────────────────────────────────────────────────────────────────────
    start_time = time.time()
    frames_processed = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break # Reached the end of the video

        # OpenCV already reads frames in BGR format, which matches the model's expectation.
        # Run your tiling logic:
        boxes = detect_tiled(
            frame, session, input_name,
            INFER_INPUT_SIZE, CLASSES_TO_MASK,
            CONF_THRESHOLD, IOU_THRESHOLD,
            width, height
        )

        # Draw solid black rectangles over detected areas (thickness=-1 means fill)
        for (bx, by, bw, bh) in boxes:
            cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (0, 0, 0), -1)

        # Write the censored frame to the new video
        out.write(frame)
        
        frames_processed += 1
        
        # Print progress every 10 frames
        if frames_processed % 10 == 0:
            elapsed = time.time() - start_time
            fps_speed = frames_processed / elapsed
            print(f"Processed {frames_processed}/{total_frames} frames ({fps_speed:.1f} frames/sec)...")

    # ──────────────────────────────────────────────────────────────────────
    # 5. Cleanup
    # ──────────────────────────────────────────────────────────────────────
    cap.release()
    out.release()
    print(f"\n[VideoBlock] Finished! Saved censored video to {OUTPUT_VIDEO}")
    print(f"Total processing time: {(time.time() - start_time):.2f} seconds.")


    # ... your video processing code ...

    print("\n[System] Video processing complete. Shutting down the PC in 60 seconds...")

    # Wait a few seconds so you can read the print statement
    time.sleep(3)

    # Send the shutdown command to Windows
    # /s = shutdown, /t 60 = wait 60 seconds before shutting down
    os.system("shutdown /s /t 60")

if __name__ == "__main__":
    main()