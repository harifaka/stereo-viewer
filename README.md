# Stereo Lab

Stereo Lab captures a pair of webcams in the browser and computes a live stereo disparity map with OpenCV.

## Run with Docker

```sh
docker compose up -d --build
```

Open <http://localhost:5000>, allow camera access, select a left and right camera, and connect. The browser accesses the host webcams; the Flask container processes the captured frames. This avoids requiring Linux `/dev/video*` devices inside Docker Desktop on Windows or macOS.

Use the disparity and block-size controls to tune the depth view. For meaningful results, use a calibrated, side-by-side stereo camera pair with matching resolution and synchronized capture.

Camera access is available to browsers on `localhost` or on HTTPS origins. If no cameras appear, grant the site camera permission and reload the page.
