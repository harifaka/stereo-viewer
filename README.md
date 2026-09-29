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

Two identical webcam models are supported. The browser lists each physical camera as a separate numbered entry (for example, `Camera 1 - USB Camera` and `Camera 2 - USB Camera`) and opens the selected device by its browser device ID. The USB vendor/product ID can be the same for both cameras; do not use it to distinguish them or add USB device passthrough to Compose.

On Windows, connect both cameras before opening the page, allow camera access in the browser and Windows privacy settings, then reload the page. If one camera cannot be opened, close other apps using it and try separate USB ports; two high-resolution streams on one USB controller can exceed its bandwidth. Camera access requires `localhost` or HTTPS.

Open **Detailed settings** to tune StereoSGBM, including minimum disparity, disparity range, block size, uniqueness and speckle filtering, left-right consistency, processing mode, and the Turbo, Viridis, or Inferno palette. Settings are applied together; canceling and reopening restores the last applied values. The disparity view scales with the available layout, and clicking either live camera feed or the disparity image opens that view full screen. Press Escape to leave full screen.

StereoSGBM generally produces a denser, more configurable disparity estimate than the previous basic StereoBM matcher. Its map is still relative disparity, not calibrated distance: accurate depth requires a rigid side-by-side stereo rig, matching image geometry, synchronized capture, and camera calibration with stereo rectification. Those calibration and rectification workflows remain future work, so tuning the matcher alone cannot correct lens distortion, camera misalignment, or timing differences.

If no cameras appear, grant the site camera permission and reload the page.
