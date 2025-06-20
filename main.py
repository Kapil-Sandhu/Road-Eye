import cv2
from ultralytics import YOLO
from screeninfo import get_monitors

# Loading model
model = YOLO('best.pt')

# Input Video
cap = cv2.VideoCapture('input_video.mp4')
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output_video.mp4', fourcc, fps, (frame_width, frame_height))

# Calculate dynamic line position based on frame dimensions
count_line_y_ratio = 0.8  # Ratio for line
count_line_position = int(frame_height * count_line_y_ratio)  # Dynamic y-position
line_start_x_ratio = 0.15  # Start x ratio
line_end_x_ratio = 0.8  # End x ratio
line_start_x = int(frame_width * line_start_x_ratio)  # Dynamic start x
line_end_x = int(frame_width * line_end_x_ratio)  # Dynamic end x

offset = 5
counter = 0
detect = []

# Define cropped region for model for faster fps
crop_height_above = int(0.3 * frame_height)  # 0.25 of frame height above line
crop_height_below = int(0.18 * frame_height)   # 0.1 of frame height below line
crop_y1 = max(0, count_line_position - crop_height_above)  # Top y-coordinate
crop_y2 = min(frame_height, count_line_position + crop_height_below)  # Bottom y-coordinate
crop_x1 = line_start_x  # Left x-coordinate (same as line start)
crop_x2 = line_end_x    # Right x-coordinate (same as line end)

# Get primary monitor resolution
try:
    monitor = get_monitors()[0]
    screen_width = monitor.width
    screen_height = monitor.height
except Exception as e:
    print(f"Could not detect screen resolution: {e}. Falling back to default 1920*1080.")
    screen_width = 1920
    screen_height = 1080

# Calculate scaling factor to fit frame within screen (preserve aspect ratio)
scale_factor = min(screen_width / frame_width, screen_height / frame_height, 1.0)
display_width = int(frame_width * scale_factor)
display_height = int(frame_height * scale_factor)

# Set up resizable window
cv2.namedWindow('Vehicle Detection', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Vehicle Detection', display_width, display_height)

def center_handle(x, y, w, h):
    # Calculate the center of a bounding box for the Count function with line
    cx = x + w // 2
    cy = y + h // 2
    return cx, cy

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Crop frame for YOLO model
    cropped_frame = frame[crop_y1:crop_y2, crop_x1:crop_x2]

    # Perform YOLO detection on cropped frame
    results = model(cropped_frame, verbose=False)  # Run model on cropped frame
    detections = results[0].boxes  # Extract detection boxes directly
    detect_data = detections.data.cpu().numpy()  # Convert to NumPy for easier handling

    for det in detect_data:
        x1, y1, x2, y2, score, class_id = det[:6]
        label = model.names[int(class_id)]

        # Filter vehicles based on class names and score threshold
        if label in ['car', 'truck', 'bus', 'motorbike'] and score > 0.2:
            # Map coordinates back to full frame
            x1_full = int(x1) + crop_x1
            y1_full = int(y1) + crop_y1
            x2_full = int(x2) + crop_x1
            y2_full = int(y2) + crop_y1

            # Draw bounding box and label on full frame
            cv2.rectangle(frame, (x1_full, y1_full), (x2_full, y2_full), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {score:.2f}", (x1_full, y1_full - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0), 2)

            # Compute center point of bounding box (in full frame coordinates)
            center = center_handle(x1_full, y1_full, x2_full - x1_full, y2_full - y1_full)
            detect.append(center)
            cv2.circle(frame, center, 4, (0, 0, 255), -1)

            # Count vehicles crossing the line
            for (cx, cy) in detect:
                if count_line_position - offset < cy < count_line_position + offset:
                    counter += 1
                    detect.remove((cx, cy))
                    print(f"Vehicle Passed: {counter}")

    # Draw count line and display counter on full frame
    cv2.line(frame, (line_start_x, count_line_position), (line_end_x, count_line_position), (255, 122, 0), 3)
    cv2.putText(frame, f"Vehicle Passed: {counter}", (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)

    # Draw yellow boundary box around cropped region
    cv2.rectangle(frame, (crop_x1, crop_y1), (crop_x2, crop_y2), (0, 255, 255), 2)  # Yellow color (BGR)

    # Resize frame for display only
    display_frame = cv2.resize(frame, (display_width, display_height))

    # Write original frame to output video
    out.write(frame)
    cv2.imshow('Vehicle Detection', display_frame)

    if cv2.waitKey(1) == 27:  # Press 'Esc' to exit
        break

cap.release()
out.release()
cv2.destroyAllWindows()
