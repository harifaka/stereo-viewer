# Stereo Vision & Multi-Sensor Fusion Project Roadmap

This document outlines the development roadmap for a Dockerized stereoscopic vision and multi-sensor fusion application using Python, OpenCV, and Flask. In the current design, the browser captures host webcams and sends frames to Flask; the container does not capture USB devices directly.

## Phase 1: Foundation & Core Stereo Vision (V1 & V2)
**Goal:** Establish a stable, containerized environment capable of handling two identical webcams without USB bandwidth crashing, providing a modern web interface for live depth perception.

- [x] **Infrastructure & Docker Setup**
  - Run Flask in Docker without camera-device passthrough; browser camera capture works with Docker Desktop on Windows and macOS.
  - Document that Windows `usbipd-win` passthrough is not required for browser-based capture.
- [x] **Browser Camera Capture & Frame Handling**
  - Enumerate two cameras independently and open each by its browser `deviceId`, including identical webcam models.
  - Send captured frames to Flask for disparity processing, keeping host camera access outside the container.
  - Keep direct V4L2/container camera capture as a separate future option if needed; it is not part of this setup.
- [x] **Stereo BM/SGBM Algorithm**
  - Convert frames to grayscale and calculate disparity with configurable `cv2.StereoSGBM_create`.
  - Filter invalid matches and visualize the disparity with a selectable Turbo, Viridis, or Inferno palette.
- [ ] **Modern Web UI (Frontend)**
  - [ ] Build UI using Bootstrap 5 (Dark Mode standard).
  - [ ] Implement Live View container using multipart HTTP stream (MJPEG).
  - [x] Create a detailed settings modal for StereoSGBM tuning via `/api/calibrate`.
  - [x] Make camera and disparity views responsive and openable in full screen.

## Phase 2: Enhanced Depth, UX, & AI Integration (V3)
**Goal:** Upgrade the depth map with interactive features, AI object detection, and better calibration techniques.

- [ ] **Automatic Checkerboard Calibration**
  - Build an admin UI to capture checkerboard patterns.
  - Implement `cv2.findChessboardCorners` and calculate camera matrices, distortion coefficients, and rectification transforms.
  - Apply stereo rectification to incoming frames before depth calculation for a much cleaner disparity map.
- [ ] **Interactive Distance Measurement**
  - Add click-event listener on the frontend video feed.
  - Send X,Y coordinates to backend.
  - Backend reads the specific disparity pixel, calculates real-world distance (Z = (focal_length * baseline) / disparity), and returns it to the UI (Virtual Tape Measure).
- [ ] **AI Object Detection with Depth (YOLOv8)**
  - Integrate `ultralytics` YOLOv8 Nano for real-time bounding box generation.
  - Extract the average or center disparity value inside the bounding box.
  - Overlay physical distance text (e.g., "Person - 2.4m") directly on the live feed.
- [ ] **3D Point Cloud Export**
  - Implement `cv2.reprojectImageTo3D` using the Q matrix.
  - Generate a `.ply` file containing X,Y,Z coordinates and RGB colors.
  - Add a frontend button to trigger generation and download the `.ply` file.
- [ ] **Depth-Based Background Blur (Bokeh)**
  - Use the disparity map as a threshold mask.
  - Apply strong Gaussian blur to pixels falling below a specific disparity threshold (background).
  - Composite the sharp foreground over the blurred background.
- [ ] **Anaglyph (Red-Cyan) 3D View**
  - Merge the Left Camera's Red channel with the Right Camera's Green and Blue channels.
  - Stream the resulting composite for viewing with standard 3D glasses.

## Phase 3: Multi-Sensor, Thermal, & Spatial Mapping (V4)
**Goal:** Move beyond basic stereo vision into industrial-grade multi-sensor fusion, supporting asymmetrical camera setups, thermal imaging, and moving environments.

- [ ] **Dynamic Multi-Camera Architecture**
  - Refactor backend to initialize 3 or 4 video streams dynamically.
  - Update UI to allow switching between different processing modes (MVS, Thermal, SLAM).
- [ ] **Multi-View Stereo (MVS) & Feature Matching**
  - Implement ORB/SIFT feature extraction for asymmetrical (non-stereo) camera placements.
  - Perform point matching (`BFMatcher` or `FlannBasedMatcher`) and draw matching lines.
  - Use Epipolar geometry to establish 3D mesh references without strict parallel alignment.
- [ ] **ChArUco Marker Global Calibration**
  - Integrate `opencv-contrib-python` for ArUco dictionary access.
  - Implement ChArUco board detection to calibrate multiple cameras looking at the same scene from vastly different angles, unifying them into a single 3D coordinate system.
- [ ] **Thermal & RGB Sensor Fusion**
  - Support secondary/tertiary inputs as thermal streams.
  - Implement Homography transformation (`cv2.findHomography` and `cv2.warpPerspective`) to align the different field-of-views and lens distortions of thermal vs RGB cameras.
  - Create a Picture-in-Picture or translucent composite overlay (Thermal mapped exactly onto RGB).
- [ ] **Thermal Stereo Vision (CLAHE)**
  - Implement Contrast Limited Adaptive Histogram Equalization (`cv2.createCLAHE`) on raw thermal images.
  - Feed the micro-contrast-enhanced thermal images into the StereoSGBM algorithm to calculate depth in pitch black or smoke-filled environments.
- [ ] **Visual SLAM & 3D Mesh Generation**
  - Implement Optical Flow (`cv2.calcOpticalFlowPyrLK`) to track corner points across consecutive frames.
  - Visualize motion vectors on the screen to track the camera rig's movement through space (Odometry).
  - (Stretch Goal) Integrate Poisson surface reconstruction to turn point clouds into solid 3D meshes suitable for game engines (Unity/Unreal).

## Tech Stack Overview
*   **Backend:** Python 3.9+, Flask, Werkzeug
*   **Computer Vision:** OpenCV (`opencv-python-headless`, `opencv-contrib-python`), NumPy
*   **AI / ML:** Ultralytics (YOLOv8)
*   **Frontend:** HTML5, JS (Fetch API), Bootstrap 5 (Dark theme)
*   **Deployment:** Docker, Docker Compose, v4l2