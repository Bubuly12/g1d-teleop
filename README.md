# XRoboToolkit Robot Teleoperation Project

The current main goal is to avoid the Vuer web page and directly use the XRoboToolkit SDK to obtain PICO headset, controller, and hand data, then drive the robot arms, Dex1 gripper, AGV base, and lift.

The current workflow supports:

- PICO controller input through `xrobotoolkit_sdk`
- Arm teleoperation through IK
- Dex1 gripper control
- AGV mobile base and lift control
- G1-D Teleimager video streaming to PICO
- Optional dataset recording

This README is organized around my current local workflow. It is an engineering usage note, not an official product document.

## 1. System Flow

Control data:

```text
PICO / XRoboToolkit
    -> xrobotoolkit_sdk
    -> teleop/xrtk/xrobotoolkit_wrapper.py
    -> TeleData
    -> teleop/teleop_hand_and_arm.py
    -> robot arm / gripper / base / lift
```

Video data:

```text
G1-D Teleimager head camera
    -> ZMQ JPEG stream
    -> TeleimagerVideoSender
    -> XRoboToolkit camera stream
    -> PICO
```

`xrobotoolkit_wrapper.py` keeps the data format as close as possible to the old `tv_wrapper.py` path, so the main teleoperation code can continue using `TeleData`.

## 2. Environment Setup

The commands below use Ubuntu 20.04 / 22.04 and a conda environment as the reference setup.

### 2.1 Create Python Environment

```bash
conda create -n xr python=3.10 pinocchio=3.1.0 casadi numpy=1.26.4 -c conda-forge
conda activate xr
```

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

Initialize the third-party submodules:

```bash
git submodule update --init --depth 1
```

### 2.2 Install Local Python Packages

Install Teleimager:

```bash
cd ~/g1d_xr/teleop/teleimager
pip install -e .
```

Install dex-retargeting:

```bash
cd ~/g1d_xr/teleop/robot_control/dex-retargeting
pip install -e .
```

Install Unitree SDK2 Python:

```bash
cd ~/g1d_xr/third_party/unitree_sdk2_python
pip install -e .
```

Install XRoboToolkit pybind SDK:

```bash
cd ~/g1d_xr/third_party/XRoboToolkit-PC-Service-Pybind
pip install -e .
```

Quick checks:

```bash
python3 -c "import xrobotoolkit_sdk; print('xrobotoolkit_sdk ok')"
python3 -c "from teleimager import ImageClient; print('teleimager ok')"
python3 -c "import pinocchio; from pinocchio import casadi as cpin; import casadi; print('ik env ok')"
```

## 3. PICO Video Sender

The video sender is built from:

```text
third_party/XRoboToolkit-Orin-Video-Sender
```

Install build dependencies:

```bash
sudo apt update
sudo apt install -y \
  build-essential cmake pkg-config \
  libopencv-dev libzmq3-dev libssl-dev \
  libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
  gstreamer1.0-tools gstreamer1.0-plugins-base \
  gstreamer1.0-plugins-good gstreamer1.0-plugins-bad \
  gstreamer1.0-plugins-ugly
```

Build:

```bash
cd ~/g1d_xr/third_party/XRoboToolkit-Orin-Video-Sender
make teleimager
```

Check H.264 plugins:

```bash
gst-inspect-1.0 x264enc
gst-inspect-1.0 h264parse
```

Standalone video test:

```bash
cd ~/g1d_xr/third_party/XRoboToolkit-Orin-Video-Sender
./TeleimagerVideoSender \
  --teleimager-host 192.168.123.164 \
  --teleimager-port 55555
```

Default video sender configuration:

- Listen address: `0.0.0.0:13579`
- Head camera source: `192.168.123.164:55555`
- Output format: `2560x720`
- Frame rate: `60 FPS`
- Encoder: `x264`

On PICO, open the XRoboToolkit Camera panel, choose `ZEDMINI` or `ZED`, click `Listen`, and enter the IP address of the computer running the video sender.

## 4. Running Teleoperation

Recommended command for the current full setup:

```bash
cd ~/g1d_xr
python3 teleop/teleop_hand_and_arm.py \
  --ee dex1 \
  --input-mode controller \
  --base-type mobile_lift \
  --headless \
  --enable-pico-video
```

Important option notes:

- `--ee dex1`: use the Dex1 gripper.
- `--input-mode controller`: use PICO controller data.
- `--base-type mobile_lift`: enable AGV base and lift control.
- `--headless`: do not create the old image client UI path.
- `--enable-pico-video`: start `TeleimagerVideoSender` together with teleoperation.

Optional recording:

```bash
python3 teleop/teleop_hand_and_arm.py \
  --ee dex1 \
  --input-mode controller \
  --base-type mobile_lift \
  --headless \
  --enable-pico-video \
  --record \
  --task-dir ./data \
  --task-name test_grasp
```

When `--record` is enabled:

- Left-controller `X`: start or stop teleoperation.
- Right-controller `A`: start recording; press again to stop and save the current episode.
- Episodes are saved under `--task-dir/--task-name`, for example `./data/test_grasp/episode_0001`.
- `data.json` stores robot states, actions, timestamps, and task metadata.
- In `--headless` mode, the old image client is disabled. PICO video is still available for live viewing, but it is not automatically saved into the episode.
- For a clean save, press `A` to stop recording first, wait until the terminal stops printing `episode_id/item_id`, and then exit the program.

If your robot computer is not `192.168.123.164`, set:

```bash
--img-server-ip <robot-ip>
```

## 5. Operation

Suggested startup order:

1. Start the robot-side Teleimager service.
2. Confirm that the host can receive images:

   ```bash
   python3 teleop/teleimager/src/teleimager/image_client.py --host 192.168.123.164
   ```

3. Start `teleop_hand_and_arm.py`.
4. In PICO XRoboToolkit, connect the Camera panel to the host computer.
5. Press the left-controller `X` button to start teleoperation.
6. Press `X` again to stop teleoperation and return the arms to the home pose.

Before enabling teleoperation, make sure the area around the robot is safe and that there are no people or obstacles inside the arm workspace.

## 6. Notes

- `--headless` only disables the old image client path. It does not disable PICO video streaming.
- PICO video uses the XRoboToolkit camera protocol and the default `13579` control port.
- The Teleimager head image is processed as left/right SBS input. The current pipeline splits the two eyes, letterboxes each eye, and then sends `2560x720` to PICO.
- The current image processing reduces distortion from direct stretching, but it is not a replacement for true stereo calibration and epipolar rectification.

## 7. Useful Test Scripts

View IK motion in PyBullet:

```bash
python3 teleop/xrtk/pybullet_xrtk_arm_ik_viewer.py
```

Test Dex1 only:

```bash
python3 teleop/robot_control/robot_hand_unitree.py --ee dex1 --xr-mode controller
```

Test mobile base and lift:

```bash
python3 teleop/robot_control/mobile_control.py --base-type mobile_lift --test-mode controller
```

## 8. Troubleshooting

If `xrobotoolkit_sdk` cannot be imported, reinstall:

```bash
cd ~/g1d_xr/third_party/XRoboToolkit-PC-Service-Pybind
pip install -e .
```

If IK import fails, check the conda packages:

```bash
python3 -c "from pinocchio import casadi as cpin; import casadi"
```

If PICO video is white or disconnected, first check whether the video sender prints that it received `OPEN_CAMERA` from the headset. Also confirm that `x264enc` and `h264parse` exist.

If the video sender reports a conda-related shared library error, run it outside conda or clear `LD_LIBRARY_PATH`. The main teleoperation launcher already tries to sanitize this when `--enable-pico-video` is used.
