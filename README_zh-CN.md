# XRoboToolkit 机器人遥操工程

这是我个人使用和维护的机器人遥操工程。当前主要目标是不用 Vuer 网页，而是直接通过 XRoboToolkit SDK 获取 PICO 的头显、手柄和手部数据，再驱动机器人机械臂、Dex1 夹爪、AGV 底盘和升降机构。

当前支持的主要功能：

- 通过 `xrobotoolkit_sdk` 获取 PICO 数据
- 机械臂 IK 遥操
- Dex1 夹爪控制
- AGV 底盘和升降控制
- G1-D Teleimager 头部视频发送到 PICO
- 可选数据采集

这份 README 按当前本地流程整理，偏工程使用说明，不是官方产品文档。

## 1. 系统流程

控制数据链路：

```text
PICO / XRoboToolkit
    -> xrobotoolkit_sdk
    -> teleop/xrtk/xrobotoolkit_wrapper.py
    -> TeleData
    -> teleop/teleop_hand_and_arm.py
    -> 机械臂 / 夹爪 / 底盘 / 升降
```

视频数据链路：

```text
G1-D Teleimager 头部相机
    -> ZMQ JPEG 图像流
    -> TeleimagerVideoSender
    -> XRoboToolkit 视频协议
    -> PICO
```

`xrobotoolkit_wrapper.py` 会尽量保持和旧版 `tv_wrapper.py` 一致的数据格式，这样主遥操代码仍然可以继续使用 `TeleData`。

## 2. 环境安装

下面以 Ubuntu 20.04 / 22.04 和 conda 环境为例。

### 2.1 创建 Python 环境

```bash
conda create -n xr python=3.10 pinocchio=3.1.0 casadi numpy=1.26.4 -c conda-forge
conda activate xr
```

安装基础 Python 依赖：

```bash
pip install -r requirements.txt
```

初始化第三方子模块：

```bash
git submodule update --init --depth 1
```

### 2.2 安装本地 Python 包

安装 Teleimager：

```bash
cd ~/xr_teleoperate/teleop/teleimager
pip install -e .
```

安装 dex-retargeting：

```bash
cd ~/xr_teleoperate/teleop/robot_control/dex-retargeting
pip install -e .
```

安装 Unitree SDK2 Python：

```bash
cd ~/xr_teleoperate/third_party/unitree_sdk2_python
pip install -e .
```

安装 XRoboToolkit pybind SDK：

```bash
cd ~/xr_teleoperate/third_party/XRoboToolkit-PC-Service-Pybind
pip install -e .
```

快速检查：

```bash
python3 -c "import xrobotoolkit_sdk; print('xrobotoolkit_sdk ok')"
python3 -c "from teleimager import ImageClient; print('teleimager ok')"
python3 -c "import pinocchio; from pinocchio import casadi as cpin; import casadi; print('ik env ok')"
```

## 3. PICO 视频发送器

视频发送器位于：

```text
third_party/XRoboToolkit-Orin-Video-Sender
```

安装编译依赖：

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

编译：

```bash
cd ~/xr_teleoperate/third_party/XRoboToolkit-Orin-Video-Sender
make teleimager
```

检查 H.264 插件：

```bash
gst-inspect-1.0 x264enc
gst-inspect-1.0 h264parse
```

单独测试视频流：

```bash
cd ~/xr_teleoperate/third_party/XRoboToolkit-Orin-Video-Sender
./TeleimagerVideoSender \
  --teleimager-host 192.168.123.164 \
  --teleimager-port 55555
```

视频发送器默认配置：

- 监听地址：`0.0.0.0:13579`
- 头部相机来源：`192.168.123.164:55555`
- 输出尺寸：`2560x720`
- 帧率：`60 FPS`
- 编码器：`x264`

PICO 端打开 XRoboToolkit 的 Camera 面板，选择 `ZEDMINI` 或 `ZED`，点击 `Listen`，输入运行视频发送器的电脑 IP 即可。

## 4. 启动遥操

当前完整流程推荐命令：

```bash
cd ~/xr_teleoperate
python3 teleop/teleop_hand_and_arm.py \
  --ee dex1 \
  --input-mode controller \
  --base-type mobile_lift \
  --headless \
  --enable-pico-video
```

重要参数说明：

- `--ee dex1`：使用 Dex1 夹爪。
- `--input-mode controller`：使用 PICO 手柄数据。
- `--base-type mobile_lift`：启用 AGV 底盘和升降。
- `--headless`：不启动旧图像客户端界面。
- `--enable-pico-video`：随主程序一起启动 PICO 视频发送器。

如果机器人端 IP 不是 `192.168.123.164`，可以指定：

```bash
--img-server-ip <robot-ip>
```

## 5. 操作顺序

推荐启动顺序：

1. 启动机器人端 Teleimager 图像服务。
2. 在本地主机确认能收到图像：

   ```bash
   python3 teleop/teleimager/src/teleimager/image_client.py --host 192.168.123.164
   ```

3. 启动 `teleop_hand_and_arm.py`。
4. 在 PICO XRoboToolkit 的 Camera 面板连接本地主机。
5. 按左手柄 `X` 键进入遥操。
6. 再按一次 `X` 键停止遥操，并让机械臂回到初始姿态。

进入遥操前，请确认机器人周围安全，机械臂工作空间内不要有人或障碍物。

## 6. 说明

- `--headless` 只是不走旧的 image client 图像路径，不会关闭 PICO 视频流。
- PICO 视频使用 XRoboToolkit 视频协议，默认控制端口是 `13579`。
- Teleimager 头部图像按左右眼 SBS 处理，目前会拆分左右眼并分别 letterbox 后发送到 PICO。
- 当前图像处理能减少直接拉伸带来的变形，但不能替代真正的双目标定和极线校正。

## 7. 常用测试脚本

查看 XRoboToolkit 原始数据：

```bash
python3 teleop/xrtk/inspect_xrtk_data.py
```

查看 wrist pose 和 IK 输出：

```bash
python3 teleop/xrtk/inspect_xrtk_arm_pose_ik.py -n 100 -t 0.2
```

用 PyBullet 看 IK 运动效果：

```bash
python3 teleop/xrtk/pybullet_xrtk_arm_ik_viewer.py
```

单独测试 Dex1：

```bash
python3 teleop/robot_control/robot_hand_unitree.py --ee dex1 --xr-mode controller
```

单独测试底盘和升降：

```bash
python3 teleop/robot_control/mobile_control.py --base-type mobile_lift --test-mode controller
```

## 8. 常见问题

如果无法导入 `xrobotoolkit_sdk`，重新安装：

```bash
cd ~/xr_teleoperate/third_party/XRoboToolkit-PC-Service-Pybind
pip install -e .
```

如果 IK 相关导入失败，检查 conda 环境：

```bash
python3 -c "from pinocchio import casadi as cpin; import casadi"
```

如果 PICO 视频白屏或断开，先看视频发送器是否收到头显发来的 `OPEN_CAMERA`。同时确认 `x264enc` 和 `h264parse` 插件存在。

如果视频发送器出现 conda 动态库相关错误，可以退出 conda 单独运行，或清理 `LD_LIBRARY_PATH`。使用 `--enable-pico-video` 时，主程序会尝试自动清理子进程环境。

