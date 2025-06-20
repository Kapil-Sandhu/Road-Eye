#RoadEye
RoadEye is a real-time vehicle detection and counting system built using the YOLO (You Only Look Once) model and OpenCV. It processes video input to detect vehicles (cars, trucks, buses, motorbikes) and counts them as they cross a dynamically positioned line, optimized for performance by processing a cropped region of the frame. The system is designed to work with videos of varying resolutions, displaying results in a scalable window and saving output with annotations.

Features

Vehicle Detection: Uses a pre-trained YOLO model to detect vehicles (cars, trucks, buses, motorbikes) with a confidence threshold of 0.4.
Dynamic Counting Line: A counting line is drawn based on video resolution, maintaining consistent positioning (e.g., 550/720 of frame height, spanning 25/1280 to 1200/1280 of frame width).
Optimized Performance: Processes a cropped region (0.25 of frame height above and 0.1 below the counting line) to improve FPS, reducing computational load.
Yellow Crop Boundary: Visualizes the processed region with a yellow bounding box.
Adaptive Display: Scales the display window to fit the user's screen resolution using screeninfo, ensuring the full frame is visible during runtime.
Output Video: Saves annotated video with detections, counting line, counter text, and crop boundary in full resolution.
Real-Time Visualization: Displays detections, counting line, vehicle count, and crop region in real-time with OpenCV.

Requirements

Python: 3.7 or higher
Dependencies:
opencv-python (OpenCV for video processing and visualization)
ultralytics (YOLO model implementation)
screeninfo (for detecting screen resolution)
numpy (included with ultralytics for array operations)



Installation

Clone the Repository:
git clone https://github.com/your-username/RoadEye.git
cd RoadEye


Set Up a Virtual Environment (recommended):
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


Install Dependencies:
pip install opencv-python ultralytics screeninfo


Download YOLO Model:

Place your pre-trained YOLO model file (best.pt) in the project root directory. Alternatively, you can train or download a YOLO model using the ultralytics library (see their documentation).


Prepare Input Video:

Place your input video file (e.g., t1.mp4) in the project root directory.



Usage

Run the Script:
python vehicle.py


Input and Output:

Input: The script processes a video file (e.g., t1.mp4) specified in vehicle.py.
Output: An annotated video (output_video.mp4) is saved with detected vehicles, a counting line, vehicle count, and a yellow crop region.
Real-Time Display: A window shows the video with annotations, scaled to fit your screen. Press Esc to exit.


Customization:

Video File: Update cap = cv2.VideoCapture('t1.mp4') in vehicle.py to use a different video.
Crop Region: Adjust crop_height_above = int(0.25 * frame_height) and crop_height_below = int(0.1 * frame_height) for a larger/smaller YOLO processing region.
Counting Line: Modify count_line_y_ratio, line_start_x_ratio, or line_end_x_ratio to reposition the counting line.



Project Structure
RoadEye/
├── vehicle.py          # Main script for vehicle detection and counting
├── best.pt            # Pre-trained YOLO model file (user-provided)
├── t1.mp4             # Input video file (user-provided)
├── output_video.mp4   # Output video with annotations (generated)
├── README.md          # Project documentation
└── DESCRIPTION.md     # Brief project description

Contributing
Contributions are welcome! To contribute:

Fork the repository.
Create a new branch (git checkout -b feature/your-feature).
Make changes and commit (git commit -m "Add your feature").
Push to the branch (git push origin feature/your-feature).
Open a pull request.

Please ensure code follows PEP 8 style guidelines and includes comments for clarity.
License
This project is licensed under the MIT License. See the LICENSE file for details.
Acknowledgments

Ultralytics YOLO for the YOLO model implementation.
OpenCV for video processing and visualization.
screeninfo for dynamic screen resolution detection.

Contact
For issues or suggestions, please open an issue on GitHub or contact [your-email@example.com].
