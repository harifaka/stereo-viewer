# Stereo Vision & Multi-Sensor Fusion Project Roadmap

This document outlines the development roadmap for a Dockerized stereoscopic vision and multi-sensor fusion application using Python, OpenCV, and Flask. In the current design, the browser captures host webcams and sends frames to Flask; the container does not capture USB devices directly. The project will scale from basic stereo depth to a fully distributed, CUDA-accelerated volumetric motion capture and 3D environment studio.

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
**Goal:** Upgrade the depth map with interactive features, AI object detection, better calibration techniques, and hardware acceleration for advanced spatial tracking.

- [x] **Automatic Checkerboard Calibration**
  - Add a settings modal with a printable, downloadable PDF and phone-displayable checkerboard.
  - Estimate camera intrinsics, stereo geometry, baseline, and rectification maps from eight or more varied views.
  - Rectify incoming frames before disparity calculation at the calibrated resolution.
- [x] **Interactive Distance Measurement**
  - Add a selectable point overlay on the live disparity map.
  - Reuse each disparity calculation to report calibrated metric distance and local disparity for the selected point.
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
- [x] **3D Pose Estimation & Spatial Tracking**
  - Integrate YOLOv8 Nano Pose to extract tracked body landmarks from the live camera feeds.
  - Cross-reference rectified joint coordinates with the disparity map and checkerboard Q matrix for metric 3D positions.
  - Add pose settings for left, right, disparity composite, or all views, plus enable and confidence controls.
- [ ] **GPU-Optimized CUDA Pipeline (Hardware Acceleration)**
  - Configure `docker-compose.yml` with NVIDIA Container Toolkit for GPU passthrough to the container.
  - Utilize a custom OpenCV build with `cv2.cuda` to keep frame data in VRAM (`cv2.cuda_GpuMat`), minimizing CPU-GPU transfer bottlenecks.
  - Run `cv2.cuda.StereoSGM` and a PyTorch/TensorRT-optimized YOLOv8 sequentially on the GPU for a high-FPS, low-CPU real-time pipeline.

## Phase 3: Multi-Sensor, Thermal, & Spatial Mapping (V4)
**Goal:** Move beyond basic stereo vision into industrial-grade multi-sensor fusion, supporting asymmetrical camera setups, thermal imaging, and moving environments.

- [ ] **Dynamic Multi-Camera Architecture**
  - [x] Add a browser-based full-window monitor for up to four distinct local camera feeds with per-camera browser-exposed controls.
  - [ ] Process three or four camera streams in the backend, beyond the existing two-camera stereo pipeline.
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

## Phase 4: Google TAP-Net & Temporal Spatial Tracking
**Goal:** Eliminate disparity flickering and stabilize tracking dynamically using TAP-Net for temporal consistency.

- [ ] **TAP-Net Integration**
  - [x] Add preparatory settings for selecting individual camera inputs and the composite output as tracking targets.
  - Deploy Google's TAP-Net in the CUDA pipeline for temporal point tracking across subsequent frames, even through occlusions.
  - Connect the target settings to live inference on selected camera streams and the composite.
- [ ] **Temporal Depth Smoothing**
  - Use TAP-Net trajectories to predict and fill in missing disparity data when objects are temporarily hidden.
  - Smooth noisy SGBM/MVS depth measurements by forcing temporal consistency on known physical points.
- [ ] **Dynamic Markerless Auto-Correction**
  - Utilize TAP-Net feature matching to detect microscopic camera shifts (e.g., thermal expansion or bumps).
  - Continuously adjust Extrinsic rotation/translation matrices on the fly without requiring the checkerboard.

## Phase 5: Distributed Camera Mesh (BYOD & IP Streams)
**Goal:** Break the physical USB limit by introducing networked mobile cameras (RTSP/WebRTC) to create a true 4+ camera 360-degree array.

- [ ] **Hybrid Network Capture Architecture**
  - Upgrade backend to handle mixed inputs simultaneously: local USB/V4L2 devices and network streams (RTSP, SRT, NDI, WebRTC).
  - Enable the use of smartphones as high-quality 4K capture nodes for specific angles.
- [ ] **Multi-Camera Triangulation (MVS Upgrade)**
  - Expand the Phase 3 ChArUco calibration to orient 4+ cameras around a central world origin `(0,0,0)`.
  - Implement Direct Linear Transformation (DLT) intersecting 2D rays from all calibrated camera angles to find exact 3D coordinates.

## Phase 6: Millisecond Synchronization & Orchestration
**Goal:** Solve temporal misalignment across mixed local/network cameras to ensure AI processes the exact same temporal slice.

- [ ] **Timestamped Ring Buffers**
  - Create dedicated threading and fixed-size `deque` buffers for each camera input stream.
  - Assign microsecond-precision timestamps to every incoming frame.
- [ ] **Master Clock & Nearest-Neighbor Sync**
  - Implement a Master Orchestrator thread that requests frames matching a specific target timestamp.
  - Extract the closest temporal match from all buffers simultaneously before dispatching to the CUDA AI pipeline.
- [ ] **Jitter & Frame-Drop Compensation**
  - Implement TAP-Net based interpolation to "guess" physical positions for a specific millisecond if a network camera drops a frame.

## Phase 7: Markerless MoCap & Unreal Engine 5 Integration
**Goal:** Translate 2D camera data into a robust 3D skeleton and animate a virtual avatar in real-time.

- [ ] **Multi-View Pose Triangulation**
  - Run YOLO-Pose simultaneously on the synchronized CUDA frames from all cameras.
  - Triangulate the 2D joint coordinates into a robust 3D skeletal structure `(X,Y,Z)`.
- [ ] **Unreal Engine Live Link**
  - Build a Python UDP server outputting OSC (Open Sound Control) or custom JSON skeletal payloads.
  - Configure UE5 to receive the skeletal data and apply it to a MetaHuman or custom avatar using Inverse Kinematics (IK).
- [ ] **Rigid Body & Prop Tracking**
  - Implement Objectron or YOLO-OBB (Oriented Bounding Boxes) to track physical props (e.g., a stick, tool, or mug).
  - Map the tracked physical prop to a corresponding virtual asset in the UE5 environment.

## Phase 8: Real-Time 3D Gaussian Splatting & WebXR ("Live 360" View)
**Goal:** Stream a fully textured, interactive 3D volumetric video of the scene directly to the browser.

- [ ] **Real-Time Point Cloud to 3DGS**
  - Implement a lightweight 3D Gaussian Splatting inference engine optimized for the 12GB VRAM limit.
  - Convert multi-camera RGB-D data into textured volumetric splats.
- [ ] **WebXR Interactive Spectator Client**
  - Replace the 2D MJPEG feed with a WebGL/Three.js spatial canvas.
  - Stream binary splat data via WebSockets to the browser, allowing viewers to orbit the live 3D scene independently of the physical camera angles.
- [ ] **Time-Freeze "Inspect" Buffer**
  - Add a UI feature for viewers to pause the live volumetric stream, freely orbit the frozen 3D moment, and resync to the live broadcast.

## Phase 9: Edu-Tech "Holographic Workshop" (Multi-Zone LOD)
**Goal:** Optimize processing for educational/crafting streams by allocating extreme detail to the workbench and ignoring irrelevant background data.

- [ ] **Static Room Pre-Scanning (Environment Map)**
  - Create a setup sequence where the user scans the empty room to generate a static 3D background, saving live GPU cycles.
- [ ] **Semantic Segmentation (Smart Green Screen)**
  - Integrate SAM (Segment Anything Model) to understand object classes in 3D.
  - Implement real-time 3D background subtraction to render messy backgrounds invisible, keeping only the presenter and the desk.
- [ ] **Asymmetric Level of Detail (LOD) Zones**
  - **Workbench Zone:** Allocate 4K network camera data and dense Gaussian Splats strictly to the crafting table for macro-level detail.
  - **Presenter Zone:** Allocate lower-res USB camera data for skeletal tracking and coarse mesh rendering of the presenter's body.

## Phase 10: Advanced Volumetric Directing & 4D Recording
**Goal:** Automate camera work using AI and enable post-production volumetric editing.

- [ ] **AI Virtual Cinematographer**
  - Create an autonomous state machine in the WebXR viewer that acts as a director.
  - Use TAP-Net and YOLO data to automatically pan, orbit, and zoom the virtual camera to follow rapid hand movements or specific actions.
- [ ] **4D Volumetric Recording**
  - Serialize and write the synchronized 3DGS and skeletal data stream to disk (`.ply` sequences or a custom binary format).
  - Create an offline viewer to replay the 4D recording, allowing the user to render standard 2D videos from entirely new, post-event camera angles.
- [ ] **Inverse Rendering (Lighting Sync)**
  - Analyze live camera feeds to determine real-world directional light and shadow intensity.
  - Transmit lighting vectors to Unreal Engine so virtual objects cast shadows matching the physical room's exact lighting setup.

---

## Tech Stack Overview
- **Backend Core:** Python 3.9+, Flask, Werkzeug, UDP/OSC (Live Link)
- **Computer Vision & Hardware Acceleration:** OpenCV (`opencv-python`, `opencv-contrib-python`), custom `cv2.cuda` build, NVIDIA Container Toolkit
- **AI / Deep Learning (TensorRT/FP16 targets):** Ultralytics (YOLOv8/Pose), Google TAP-Net, Segment Anything Model (SAM), PyTorch
- **Multi-Camera Sync:** RTSP/NDI/WebRTC streaming protocols, Timestamped deque buffers
- **3D & Volumetric Mapping:** 3D Gaussian Splatting, Multi-View Stereo (MVS), Direct Linear Transformation (DLT), Visual SLAM
- **Frontend & WebXR:** HTML5, JS (Fetch API/WebSockets), Bootstrap 5 (Dark theme), Three.js / WebGL
- **Game Engine Integration:** Unreal Engine 5 (MetaHuman, Live Link, Inverse Kinematics)
- **Deployment:** Docker, Docker Compose, v4l2