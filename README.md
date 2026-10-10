# Stereo Lab

Stereo Lab captures a pair of webcams in the browser and computes a live stereo disparity map with OpenCV.

## Run with Docker

```sh
docker compose up -d --build
```

Follow application and camera diagnostics with:

```sh
docker compose logs -f stereo_app
```

Browser camera events appear as `browser_camera` entries. They include permission, device enumeration, each camera open result, track state, and browser error details. Frame upload and OpenCV processing results appear as `disparity` entries. Camera device IDs are not written to the logs.

Open <http://localhost:5000>. The stereo monitor captures its selected pair in the browser and sends frames to Flask for processing. The multi-camera monitor supports browser-local capture and direct Linux/Docker capture; choose its source with the gear button beside the language selector.

The interface is available in English and Hungarian. Use the **EN / HU** selector in the page header; the selected language is remembered in this browser.

Camera source is a browser-local preference, so remote users can select **Browser cameras** to use webcams attached to their own computers while local users select **Docker / Linux devices** to use cameras attached to the server. A browser's source choice is remembered only in that browser; camera slot labels, indexes, and Active checkboxes are shared by the server.

The multi-camera monitor uses 1 to 16 slots. Device indexes are unique integers from 0 to 31. A fresh install starts with four slots and the first two active. Existing four-slot `cameras.json` files still load.

In browser mode, an index is the position in the video-device list reported to that user's browser. Grant webcam permission and use `localhost` or HTTPS; remote users need HTTPS for browser webcam access. The browser's device IDs are not written to server configuration. Browser index choices are stored in that browser and follow the current slot count.

In Docker/Linux mode, index N opens `/dev/videoN` inside the container. The included Compose file maps `/dev/video0` through `/dev/video3` and persists the shared camera mapping at `./camera-config/cameras.json`. Every Active camera is read concurrently by its own server capture worker and its MJPEG preview is available to clients. This exposes server-attached cameras to every user who can access the app; use appropriate network access controls when deploying it remotely.

To use a Linux camera above index 3, add a matching Compose `devices` entry, for example `"/dev/video4:/dev/video4"`. Add an entry only when that device node exists; mapping a missing node prevents Compose from starting.

On Windows 11 with WSL2, Docker/Linux mode works only when the webcams have first been made available as `/dev/video*` devices to the Linux environment used by Docker. If Windows does not expose them there, use Browser cameras; browser capture does not require USB passthrough.

Open **Detailed settings** to tune StereoSGBM when the depth map is too noisy, misses useful detail, or needs a different search span or display palette. Hover over a setting for a short explanation of its effect. With both cameras connected, select **Start preview** to see a separate live disparity preview calculated from the draft values; it does not change the main depth view or the active settings. Adjustments update the preview as it runs. Choose **Apply settings** to save the current values, or **Cancel**, **Close**, or Escape to discard the draft and restore the last applied values. The disparity view scales with the available layout, and clicking either live camera feed or the disparity image opens that view full screen. Press Escape to leave full screen.

### Multi-Camera Monitor

Open **Multi-camera monitor** from the shared **Pages** panel on either page for a live preview of the configured cameras. One camera fills the page; additional cameras wrap across the grid. Use the gear button to select the capture source, add or remove slots (1 to 16), and map each slot to an index, custom label, and Active state. Press **Connect active cameras** to start enabled feeds or **Disconnect cameras** to release them. The shared settings are atomically saved to `config/cameras.json` in the container and persisted to the host `./camera-config` folder by Compose.

Choose a processing mode under the grid: **Preview**, **Multi-view stereo**, **Thermal fusion**, **Thermal stereo**, or **Visual SLAM**. Preview only shows the live grid. The other modes add a server-rendered output. Browser mode uploads frames from this computer while a processing mode is active. Docker mode processes the frames already captured on the server. Multi-view stereo and thermal stereo need two camera frames and show an error when fewer are available. Preview and visual SLAM still run with one camera. Thermal fusion needs one RGB camera and at least one other camera marked as thermal.

Multi-view stereo matches ORB or SIFT features against a reference camera, draws inlier matches and epipolar lines, and triangulates a sparse 3D reference. It is not a dense mesh. Thermal fusion estimates a homography and overlays a translucent thermal colormap on the RGB view, with an optional picture-in-picture. Thermal stereo applies CLAHE and then StereoSGBM; the disparity colors follow the stereo-page palette. Metric distance still requires checkerboard calibration on the stereo page. Visual SLAM draws optical-flow vectors and a relative path from the selected odometry camera. Processing settings, flow state, and calibration are held in memory and reset when the app restarts.

**ChArUco global calibration** prints or displays a board from **Open board**. Show it to at least two cameras, capture four or more varied views, and choose **Calibrate cameras**. The reference camera becomes the origin of one coordinate system. OpenCV ArUco support comes from `opencv-contrib-python`. Poisson surface reconstruction and network camera streams are not implemented.

The **TAP-Net temporal tracking** section saves preparatory target selections for individual camera inputs and the composite output. The TAP-Net model/runtime is not integrated yet, so saving these settings does not run tracking. Configuration is held by the running Flask process and resets when it restarts.

### Camera Hardware Controls

Open **Detailed settings** > **Camera hardware** while the stereo pair is connected. Choose the left or right input, then select a discovered camera device to switch that input without stopping the other stream. The panel reads each active video track's browser-reported capabilities and creates controls for the options that camera and browser expose, including supported focus/white-balance/exposure modes, numeric controls such as focus distance, zoom, brightness, and frame rate, and supported width/height values. Changes apply immediately to the selected camera; **Reset device controls** asks the browser to restore its defaults for that track.

These are Web Media Capture controls, not direct USB or manufacturer-driver access. Available controls depend on the webcam model, operating system, browser, and driver; autofocus, depth settings, manual exposure, and other controls may be absent or rejected by a device. A browser device ID is shown for identification and can be selected from the discovered-device list, but it is opaque, privacy-scoped, and cannot be edited into a USB serial number, vendor/product ID, or arbitrary hardware ID. Vendor-specific settings require the camera manufacturer's software or a native application. Stereo Lab calculates depth from the two video streams, so a camera's proprietary depth API is only usable if the browser exposes it as a supported video capability.

Camera hardware controls on the stereo monitor use browser Web Media Capture. The multi-camera monitor's Docker/Linux source instead uses OpenCV to read the mapped video devices inside the container; it does not expose browser hardware controls.

## Camera Calibration and Measurement

Connect both cameras, then open **Detailed settings** > **Camera calibration**. The default checkerboard has 9 x 6 inner corners (10 x 7 squares). Use **Download PDF** for a vector target with the configured physical square size, or **Print target** to open the SVG. Print at actual size; the default square is 18 mm. **Full-screen target** enlarges the checkerboard in the app. You can also open the phone target URL on a phone and show it full-screen to both cameras. For a screen target, measure one displayed square with a ruler and enter that physical size in **Square size (mm)**. For phone access, replace `localhost` in the URL with the PC's local network IP address and keep both devices on the same Wi-Fi network.

For target-free setup, choose **Feature alignment (no target)** instead. Keep the stereo rig fixed, point both cameras at the same detailed scene, and choose **Align from current pair**. OpenCV enhances local contrast with CLAHE, matches SIFT features, rejects inconsistent correspondences with RANSAC, then estimates uncalibrated stereo rectification. This can align the images and provide relative disparity without a checkerboard; it cannot determine camera intrinsics or real-world scale, so metric point-distance measurements remain unavailable. It needs shared textured areas and enough visible detail; CLAHE may help dim images, but cannot recover detail lost to darkness, blur, or noise. Keep the cameras fixed and recalibrate if they move.

Feature matching currently runs on the CPU. The `opencv-python-headless` wheel used by this project does not include CUDA support, and enabling Docker's NVIDIA runtime alone does not make these OpenCV operations use the GPU. GPU acceleration would require a CUDA-enabled OpenCV build and compatible container/runtime configuration.

Start camera capture and open calibration to see both live camera views and the live stereo disparity composite. Keep the complete, flat checkerboard visible in both cameras. If automatic detection misses it, drag a box around the full board in each camera preview and retry; the app searches the full image as a fallback. Move and tilt the board through varied positions and distances, capturing each useful pose with **Capture board view**. The app needs eight accepted paired views before it estimates both cameras, stereo geometry, and rectification. Small fixed height or angle differences between cameras are handled by stereo calibration; keep the camera rig fixed while collecting views and during later measurement. Calibration applies to the captured camera resolution; recalibrate if the resolution changes. Captured views and calibration are held in app memory and are cleared when the app container restarts. **Reset calibration** clears the current model and captured views.

After calibration, click a point on the live **Disparity map**. A marker appears and the metric distance and local disparity refresh with each processed frame. The estimate uses a small neighborhood around the selected pixel; textureless, occluded, reflective, or otherwise unmatched areas may have no reliable result. A valid metric estimate depends on a rigid, synchronized stereo pair, the correct checkerboard cell size, and varied calibration views. It is an estimate, not a substitute for a depth sensor or a verified measurement instrument.

### Pose Estimation and Spatial Tracking

Open **Detailed settings** > **Pose tracking** to enable YOLOv8 Nano Pose, choose a confidence threshold, and display the skeleton on the left camera, right camera, disparity composite, or all three views. The model downloads automatically the first time pose estimation is enabled; the container needs internet access for that first download. The Docker image uses CPU-only PyTorch and does not include CUDA runtime support. Pose inference runs on the CPU and reduces the available frame rate, especially when displaying both camera feeds. The model is loaded only when pose estimation is first used.

Pose landmarks provide 2D tracking without calibration. Metric 3D joint coordinates and distance labels require checkerboard stereo calibration at the current camera resolution; feature alignment alone does not provide real-world scale. Each landmark uses a small local disparity neighborhood, so joints on occluded, textureless, or mismatched regions may not have a 3D estimate. Right-camera landmarks are transformed into rectified stereo coordinates before depth lookup. This tracking output is a visual estimate and should not be used as a safety-rated measurement or control input.

StereoSGBM calculates relative disparity. Stereo rectification corrects lens distortion and camera alignment using the calibration, while the known checkerboard scale and measured baseline allow the app to estimate metric distance. Matcher tuning alone cannot correct a moving rig, unsynchronized cameras, poor calibration views, or an incorrectly measured target.

If no cameras appear, grant the site camera permission and reload the page.
