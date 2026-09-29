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

Open <http://localhost:5000>, allow camera access, select a left and right camera, and connect. The browser accesses the host webcams and sends captured frames to the Flask container for processing. Docker does not need direct USB or `/dev/video*` access; this is the supported setup for Docker Desktop on Windows and macOS.

The interface is available in English and Hungarian. Use the **EN / HU** selector in the page header; the selected language is remembered in this browser.

Two identical webcam models are supported. The browser lists each physical camera as a separate numbered entry (for example, `Camera 1 - USB Camera` and `Camera 2 - USB Camera`) and opens the selected device by its browser device ID. The USB vendor/product ID can be the same for both cameras; do not use it to distinguish them or add USB device passthrough to Compose.

On Windows, connect both cameras before opening the page, allow camera access in the browser and Windows privacy settings, then reload the page. If one camera cannot be opened, close other apps using it and try separate USB ports; two high-resolution streams on one USB controller can exceed its bandwidth. Camera access requires `localhost` or HTTPS.

Open **Detailed settings** to tune StereoSGBM when the depth map is too noisy, misses useful detail, or needs a different search span or display palette. Hover over a setting for a short explanation of its effect. With both cameras connected, select **Start preview** to see a separate live disparity preview calculated from the draft values; it does not change the main depth view or the active settings. Adjustments update the preview as it runs. Choose **Apply settings** to save the current values, or **Cancel**, **Close**, or Escape to discard the draft and restore the last applied values. The disparity view scales with the available layout, and clicking either live camera feed or the disparity image opens that view full screen. Press Escape to leave full screen.

### Camera Hardware Controls

Open **Detailed settings** > **Camera hardware** while the stereo pair is connected. Choose the left or right input, then select a discovered camera device to switch that input without stopping the other stream. The panel reads each active video track's browser-reported capabilities and creates controls for the options that camera and browser expose, including supported focus/white-balance/exposure modes, numeric controls such as focus distance, zoom, brightness, and frame rate, and supported width/height values. Changes apply immediately to the selected camera; **Reset device controls** asks the browser to restore its defaults for that track.

These are Web Media Capture controls, not direct USB or manufacturer-driver access. Available controls depend on the webcam model, operating system, browser, and driver; autofocus, depth settings, manual exposure, and other controls may be absent or rejected by a device. A browser device ID is shown for identification and can be selected from the discovered-device list, but it is opaque, privacy-scoped, and cannot be edited into a USB serial number, vendor/product ID, or arbitrary hardware ID. Vendor-specific settings require the camera manufacturer's software or a native application. Stereo Lab calculates depth from the two video streams, so a camera's proprietary depth API is only usable if the browser exposes it as a supported video capability.

In Docker, the browser on the host still owns and configures the USB cameras, then sends captured frames to the container for stereo processing. No `/dev/video*` mapping or USB passthrough is needed. Open the site at `localhost` or HTTPS and grant camera permission; the container cannot add controls that the browser does not expose.

## Camera Calibration and Measurement

Connect both cameras, then open **Detailed settings** > **Camera calibration**. The default checkerboard has 9 x 6 inner corners (10 x 7 squares). Use **Download PDF** for a vector target with the configured physical square size, or **Print target** to open the SVG. Print at actual size; the default square is 18 mm. **Full-screen target** enlarges the checkerboard in the app. You can also open the phone target URL on a phone and show it full-screen to both cameras. For a screen target, measure one displayed square with a ruler and enter that physical size in **Square size (mm)**. For phone access, replace `localhost` in the URL with the PC's local network IP address and keep both devices on the same Wi-Fi network.

Start camera capture and open calibration to see both live camera views and the live stereo disparity composite. Keep the complete, flat checkerboard visible in both cameras. If automatic detection misses it, drag a box around the full board in each camera preview and retry; the app searches the full image as a fallback. Move and tilt the board through varied positions and distances, capturing each useful pose with **Capture board view**. The app needs eight accepted paired views before it estimates both cameras, stereo geometry, and rectification. Small fixed height or angle differences between cameras are handled by stereo calibration; keep the camera rig fixed while collecting views and during later measurement. Calibration applies to the captured camera resolution; recalibrate if the resolution changes. Captured views and calibration are held in app memory and are cleared when the app container restarts. **Reset calibration** clears the current model and captured views.

After calibration, click a point on the live **Disparity map**. A marker appears and the metric distance and local disparity refresh with each processed frame. The estimate uses a small neighborhood around the selected pixel; textureless, occluded, reflective, or otherwise unmatched areas may have no reliable result. A valid metric estimate depends on a rigid, synchronized stereo pair, the correct checkerboard cell size, and varied calibration views. It is an estimate, not a substitute for a depth sensor or a verified measurement instrument.

StereoSGBM calculates relative disparity. Stereo rectification corrects lens distortion and camera alignment using the calibration, while the known checkerboard scale and measured baseline allow the app to estimate metric distance. Matcher tuning alone cannot correct a moving rig, unsynchronized cameras, poor calibration views, or an incorrectly measured target.

If no cameras appear, grant the site camera permission and reload the page.
