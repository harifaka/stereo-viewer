# Stereo Lab

Stereo Lab captures a pair of webcams in the browser and computes a live stereo disparity map with OpenCV.

## Run with Docker

```sh
docker compose up -d --build
```

Open <http://localhost:5000>, allow camera access, select a left and right camera, and connect. The browser accesses the host webcams and sends captured frames to the Flask container for processing. Docker does not need direct USB or `/dev/video*` access; this is the supported setup for Docker Desktop on Windows and macOS.

Two identical webcam models are supported. The browser lists each physical camera as a separate numbered entry (for example, `Camera 1 - USB Camera` and `Camera 2 - USB Camera`) and opens the selected device by its browser device ID. The USB vendor/product ID can be the same for both cameras; do not use it to distinguish them or add USB device passthrough to Compose.

On Windows, connect both cameras before opening the page, allow camera access in the browser and Windows privacy settings, then reload the page. If one camera cannot be opened, close other apps using it and try separate USB ports; two high-resolution streams on one USB controller can exceed its bandwidth. Camera access requires `localhost` or HTTPS.

Use the disparity and block-size controls to tune the depth view. For meaningful results, use a calibrated, side-by-side stereo camera pair with matching resolution and synchronized capture.

If no cameras appear, grant the site camera permission and reload the page.
