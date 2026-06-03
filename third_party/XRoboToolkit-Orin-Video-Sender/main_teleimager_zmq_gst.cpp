#include <chrono>
#include <csignal>
#include <cstring>
#include <gst/app/gstappsink.h>
#include <gst/app/gstappsrc.h>
#include <gst/gst.h>
#include <iomanip>
#include <iostream>
#include <memory>
#include <opencv2/opencv.hpp>
#include <stdexcept>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#include <zmq.h>

#include "network_asio.hpp"

std::unique_ptr<TCPClient> sender_ptr;
std::unique_ptr<TCPServer> server_ptr;
volatile sig_atomic_t stop_requested = 0;

bool send_enabled = false;
bool encoding_enabled = false;
bool send_over_listen_socket = false;
bool auto_send_on_client = false;
std::string send_to_server = "127.0.0.1";
int send_to_port = 12345;
int encoded_frame_count = 0;
int pushed_frame_count = 0;
std::vector<uint8_t> control_rx_buffer;

struct CameraRequestData {
  int width;
  int height;
  int fps;
  int bitrate;
  int enable_mv_hevc;
  int render_mode;
  int port;
  std::string camera;
  std::string ip;

  CameraRequestData()
      : width(0), height(0), fps(0), bitrate(0), enable_mv_hevc(0),
        render_mode(0), port(0) {}
};

struct NetworkDataProtocol {
  std::string command;
  std::vector<uint8_t> data;
};

std::string printTimeMs(const std::string &tag, bool force_print = true) {
  auto now = std::chrono::system_clock::now();
  auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                    now.time_since_epoch())
                    .count();
  std::stringstream ss;
  ss << "===LATENCY TEST===[" << tag << "]\t" << now_ms << " ms\n";
  if (force_print) {
    std::cout << ss.str();
  }
  return ss.str();
}

bool initialize_sender() {
  int retry = 10;
  while (retry > 0 && !sender_ptr) {
    try {
      sender_ptr.reset(new TCPClient(send_to_server, send_to_port));
      std::cout << "Attempting to connect to " << send_to_server << ":"
                << send_to_port << std::endl;
      sender_ptr->connect();
      return true;
    } catch (const TCPException &e) {
      std::cerr << "Failed to connect to server: " << e.what() << std::endl;
      sender_ptr = nullptr;
    }
    std::this_thread::sleep_for(std::chrono::seconds(1));
    retry--;
  }
  return false;
}

int32_t readInt32LE(const std::vector<uint8_t> &data, size_t offset) {
  if (offset + 4 > data.size()) {
    throw std::out_of_range("Not enough data to read int32");
  }
  return static_cast<int32_t>((data[offset]) | (data[offset + 1] << 8) |
                              (data[offset + 2] << 16) |
                              (data[offset + 3] << 24));
}

std::string readCompactString(const std::vector<uint8_t> &data,
                              size_t &offset) {
  if (offset >= data.size()) {
    throw std::out_of_range("Not enough data to read string length");
  }

  const uint8_t length = data[offset++];
  if (length == 0) {
    return std::string();
  }
  if (offset + length > data.size()) {
    throw std::out_of_range("Not enough data to read string content");
  }

  std::string result(reinterpret_cast<const char *>(&data[offset]), length);
  offset += length;
  return result;
}

NetworkDataProtocol deserializeNetworkProtocol(
    const std::vector<uint8_t> &buffer) {
  if (buffer.size() < 8) {
    throw std::invalid_argument("Buffer too small for protocol data");
  }

  size_t offset = 0;
  const int32_t command_length = readInt32LE(buffer, offset);
  offset += 4;
  if (command_length < 0 || offset + command_length > buffer.size()) {
    throw std::invalid_argument("Invalid command length");
  }

  std::string command;
  if (command_length > 0) {
    command =
        std::string(reinterpret_cast<const char *>(&buffer[offset]),
                    command_length);
    const size_t null_pos = command.find('\0');
    if (null_pos != std::string::npos) {
      command = command.substr(0, null_pos);
    }
  }
  offset += command_length;

  const int32_t data_length = readInt32LE(buffer, offset);
  offset += 4;
  if (data_length < 0 || offset + data_length > buffer.size()) {
    throw std::invalid_argument("Invalid data length");
  }

  NetworkDataProtocol protocol;
  protocol.command = command;
  if (data_length > 0) {
    protocol.data.assign(buffer.begin() + offset,
                         buffer.begin() + offset + data_length);
  }
  return protocol;
}

CameraRequestData deserializeCameraRequest(const std::vector<uint8_t> &data) {
  if (data.size() < 31) {
    throw std::invalid_argument("Data is too small for camera request");
  }

  size_t offset = 0;
  if (data[offset] != 0xCA || data[offset + 1] != 0xFE) {
    throw std::invalid_argument("Invalid camera request magic bytes");
  }
  offset += 2;

  const uint8_t version = data[offset++];
  if (version != 1) {
    throw std::invalid_argument("Unsupported camera request version");
  }

  CameraRequestData result;
  result.width = readInt32LE(data, offset);
  result.height = readInt32LE(data, offset + 4);
  result.fps = readInt32LE(data, offset + 8);
  result.bitrate = readInt32LE(data, offset + 12);
  result.enable_mv_hevc = readInt32LE(data, offset + 16);
  result.render_mode = readInt32LE(data, offset + 20);
  result.port = readInt32LE(data, offset + 24);
  offset += 28;

  result.camera = readCompactString(data, offset);
  result.ip = readCompactString(data, offset);
  return result;
}

bool tryHandleOpenCameraProtocol(const std::string &raw_command) {
  std::vector<uint8_t> binary(raw_command.begin(), raw_command.end());
  if (binary.size() < 4) {
    return false;
  }

  const uint32_t body_length =
      (static_cast<uint32_t>(binary[0]) << 24) |
      (static_cast<uint32_t>(binary[1]) << 16) |
      (static_cast<uint32_t>(binary[2]) << 8) |
      static_cast<uint32_t>(binary[3]);
  if (body_length == 0 || 4 + body_length > binary.size()) {
    return false;
  }

  std::vector<uint8_t> body(binary.begin() + 4,
                            binary.begin() + 4 + body_length);
  NetworkDataProtocol protocol = deserializeNetworkProtocol(body);
  std::cout << "[Pico Ctrl] command = " << protocol.command << std::endl;

  if (protocol.command == "CLOSE_CAMERA") {
    encoding_enabled = false;
    send_enabled = false;
    if (sender_ptr) {
      sender_ptr->disconnect();
      sender_ptr = nullptr;
    }
    std::cout << "Closed Teleimager camera stream" << std::endl;
    return true;
  }

  if (protocol.command != "OPEN_CAMERA") {
    return true;
  }

  CameraRequestData camera_config = deserializeCameraRequest(protocol.data);
  std::cout << "Camera config from headset - width=" << camera_config.width
            << ", height=" << camera_config.height
            << ", fps=" << camera_config.fps
            << ", bitrate=" << camera_config.bitrate
            << ", camera=" << camera_config.camera
            << ", target=" << camera_config.ip << ":" << camera_config.port
            << std::endl;

  if (camera_config.ip.empty() || camera_config.port <= 0) {
    std::cerr << "OPEN_CAMERA did not provide a valid target ip/port"
              << std::endl;
    return true;
  }

  send_to_server = camera_config.ip;
  send_to_port = camera_config.port;
  if (sender_ptr) {
    sender_ptr->disconnect();
    sender_ptr = nullptr;
  }

  if (initialize_sender()) {
    encoding_enabled = true;
    send_enabled = true;
    std::cout << "Started Teleimager stream from OPEN_CAMERA protocol"
              << std::endl;
  } else {
    stop_requested = 1;
  }
  return true;
}

void printErrorAndQuit(const std::string &error_msg) {
  std::cerr << "Error: " << error_msg << std::endl;
  stop_requested = 1;
}

void handleSimpleControlCommand(const std::string &command) {
  if (command == "StartRobotCameraStream") {
    if (initialize_sender()) {
      encoding_enabled = true;
      send_enabled = true;
      std::cout << "Started Teleimager encoding and sending" << std::endl;
    } else {
      stop_requested = 1;
    }
  } else if (command == "StopRobotCameraStream") {
    encoding_enabled = false;
    send_enabled = false;
    std::cout << "Stopped Teleimager encoding and sending" << std::endl;
  } else if (command.substr(0, 8) == "LOOPTEST") {
    printTimeMs("Loop Receive");
  } else if (command.substr(0, 12) == "MediaDecoder") {
    printTimeMs("Java - " + command);
  } else {
    std::cerr << "Unknown command received: " << command << std::endl;
  }
}

bool tryConsumeWrappedControlPackets() {
  bool consumed_any = false;
  const size_t max_body_size = 10 * 1024 * 1024;

  while (control_rx_buffer.size() >= 4) {
    const uint32_t body_length =
        (static_cast<uint32_t>(control_rx_buffer[0]) << 24) |
        (static_cast<uint32_t>(control_rx_buffer[1]) << 16) |
        (static_cast<uint32_t>(control_rx_buffer[2]) << 8) |
        static_cast<uint32_t>(control_rx_buffer[3]);

    if (body_length == 0 || body_length > max_body_size) {
      return consumed_any;
    }

    const size_t packet_size = 4 + static_cast<size_t>(body_length);
    if (control_rx_buffer.size() < packet_size) {
      std::cout << "Waiting for remaining OPEN_CAMERA packet bytes: have "
                << control_rx_buffer.size() << ", need " << packet_size
                << std::endl;
      return consumed_any;
    }

    std::string raw_packet(control_rx_buffer.begin(),
                           control_rx_buffer.begin() + packet_size);
    control_rx_buffer.erase(control_rx_buffer.begin(),
                            control_rx_buffer.begin() + packet_size);

    try {
      if (tryHandleOpenCameraProtocol(raw_packet)) {
        consumed_any = true;
      }
    } catch (const std::exception &e) {
      std::cerr << "Failed to parse wrapped control packet: " << e.what()
                << std::endl;
    }
  }

  return consumed_any;
}

void onDataCallback(const std::string &command) {
  std::cout << "[Pico Ctrl] recv bytes = " << command.size()
            << ", first bytes=";
  for (size_t i = 0; i < std::min(command.size(), size_t(16)); ++i) {
    std::cout << std::hex << std::setfill('0') << std::setw(2)
              << static_cast<unsigned int>(
                     static_cast<unsigned char>(command[i]))
              << " ";
  }
  std::cout << std::dec << std::endl;

  if (command == "StartRobotCameraStream" ||
      command == "StopRobotCameraStream" ||
      command.substr(0, 8) == "LOOPTEST" ||
      command.substr(0, 12) == "MediaDecoder") {
    handleSimpleControlCommand(command);
    return;
  }

  control_rx_buffer.insert(control_rx_buffer.end(), command.begin(),
                           command.end());
  if (tryConsumeWrappedControlPackets()) {
    return;
  }

  if (control_rx_buffer.size() > 4) {
    const uint32_t possible_body_length =
        (static_cast<uint32_t>(control_rx_buffer[0]) << 24) |
        (static_cast<uint32_t>(control_rx_buffer[1]) << 16) |
        (static_cast<uint32_t>(control_rx_buffer[2]) << 8) |
        static_cast<uint32_t>(control_rx_buffer[3]);
    if (possible_body_length == 0 ||
        possible_body_length > 10 * 1024 * 1024) {
      std::string fallback(control_rx_buffer.begin(), control_rx_buffer.end());
      control_rx_buffer.clear();
      std::cout << "Control command received: " << fallback << std::endl;
      handleSimpleControlCommand(fallback);
    }
  }
}

void onDisconnectCallback() {
  std::cout << "onDisconnectCallback" << std::endl;
  encoding_enabled = false;
  send_enabled = false;
  if (sender_ptr) {
    sender_ptr->disconnect();
    sender_ptr = nullptr;
  }
}

void handle_sigint(int) {
  std::cout << "\nSIGINT received. Stopping ..." << std::endl;
  if (server_ptr) {
    server_ptr->stop();
    server_ptr = nullptr;
  }
  if (sender_ptr) {
    sender_ptr->disconnect();
    sender_ptr = nullptr;
  }
  stop_requested = 1;
}

GstFlowReturn on_new_sample(GstAppSink *sink, gpointer) {
  GstSample *sample = gst_app_sink_pull_sample(sink);
  if (!sample) {
    return GST_FLOW_ERROR;
  }

  GstBuffer *buffer = gst_sample_get_buffer(sample);
  GstMapInfo map;
  if (gst_buffer_map(buffer, &map, GST_MAP_READ)) {
    const uint8_t *data = map.data;
    gsize size = map.size;
    encoded_frame_count++;
    if (encoded_frame_count % 30 == 1) {
      std::cout << "Encoded H.264 sample " << encoded_frame_count
                << ", bytes=" << size
                << ", send_enabled=" << send_enabled
                << ", tcp_connected="
                << (sender_ptr && sender_ptr->isConnected()) << std::endl;
    }

    if (send_enabled && data && size > 0) {
      try {
        std::vector<uint8_t> packet(4 + size);
        packet[0] = (size >> 24) & 0xFF;
        packet[1] = (size >> 16) & 0xFF;
        packet[2] = (size >> 8) & 0xFF;
        packet[3] = size & 0xFF;
        std::copy(data, data + size, packet.begin() + 4);

        if (sender_ptr && sender_ptr->isConnected()) {
          sender_ptr->sendData(packet);
        } else if (send_over_listen_socket && server_ptr &&
                   server_ptr->isClientConnected()) {
          server_ptr->sendData(packet);
        } else {
          gst_buffer_unmap(buffer, &map);
          gst_sample_unref(sample);
          return GST_FLOW_OK;
        }

        static int sent_frame_count = 0;
        if (sent_frame_count % 30 == 0) {
          std::cout << "Sent " << size << " bytes of H.264 data" << std::endl;
        }
        sent_frame_count++;
      } catch (const TCPException &e) {
        printErrorAndQuit(e.what());
      } catch (const std::exception &e) {
        printErrorAndQuit("Unexpected error during sendData: " +
                          std::string(e.what()));
      }
    }
    gst_buffer_unmap(buffer, &map);
  }

  gst_sample_unref(sample);
  return GST_FLOW_OK;
}

std::string buildPipelineString(const std::string &encoder, int width, int height,
                                int fps, int bitrate, bool preview_enabled) {
  std::stringstream ss;
  ss << "appsrc name=mysource is-live=true format=time do-timestamp=true "
     << "block=false "
     << "caps=video/x-raw,format=BGRA,width=" << width << ",height=" << height
     << ",framerate=" << fps << "/1 ! ";

  if (preview_enabled) {
    ss << "tee name=t "
       << "t. ! queue leaky=downstream max-size-buffers=1 ! ";
  } else {
    ss << "queue leaky=downstream max-size-buffers=1 ! ";
  }

  if (encoder == "x264") {
    ss << "videoconvert ! video/x-raw,format=I420 ! "
       << "x264enc tune=zerolatency speed-preset=ultrafast key-int-max=15 "
       << "bitrate=" << bitrate / 1000 << " byte-stream=true ! "
       << "h264parse config-interval=-1 ! "
       << "video/x-h264,stream-format=byte-stream,alignment=au ! ";
  } else {
    ss << "videoconvert ! nvvidconv ! video/x-raw(memory:NVMM),format=NV12 ! "
       << "nvv4l2h264enc maxperf-enable=1 insert-sps-pps=true "
       << "idrinterval=15 bitrate=" << bitrate << " ! "
       << "h264parse config-interval=-1 ! "
       << "video/x-h264,stream-format=byte-stream,alignment=au ! ";
  }

  ss << "appsink name=mysink emit-signals=true sync=false async=false "
     << "max-buffers=1 drop=true ";

  if (preview_enabled) {
    ss << "t. ! queue leaky=downstream max-size-buffers=1 ! "
       << "videoconvert ! autovideosink sync=false ";
  }

  return ss.str();
}

cv::Mat decodeJpegToBgr(const std::vector<uint8_t> &jpg) {
  cv::Mat encoded(1, static_cast<int>(jpg.size()), CV_8UC1,
                  const_cast<uint8_t *>(jpg.data()));
  cv::Mat bgr = cv::imdecode(encoded, cv::IMREAD_COLOR);
  return bgr;
}

cv::Mat letterboxToSize(const cv::Mat &input, int target_width,
                        int target_height) {
  if (input.empty() || target_width <= 0 || target_height <= 0) {
    return cv::Mat();
  }

  const double scale = std::min(static_cast<double>(target_width) / input.cols,
                                static_cast<double>(target_height) / input.rows);
  const int resized_width = std::max(1, static_cast<int>(input.cols * scale));
  const int resized_height = std::max(1, static_cast<int>(input.rows * scale));

  cv::Mat resized;
  cv::resize(input, resized, cv::Size(resized_width, resized_height), 0, 0,
             cv::INTER_LINEAR);

  cv::Mat canvas(target_height, target_width, input.type(), cv::Scalar::all(0));
  const int x0 = (target_width - resized_width) / 2;
  const int y0 = (target_height - resized_height) / 2;
  resized.copyTo(canvas(cv::Rect(x0, y0, resized_width, resized_height)));
  return canvas;
}

cv::Mat prepareStereoSbsFrame(const cv::Mat &input_bgr, int width, int height) {
  if (input_bgr.empty() || input_bgr.cols % 2 != 0 || width % 2 != 0) {
    return cv::Mat();
  }

  const int input_eye_width = input_bgr.cols / 2;
  const int output_eye_width = width / 2;

  cv::Mat left_raw = input_bgr(cv::Rect(0, 0, input_eye_width, input_bgr.rows));
  cv::Mat right_raw =
      input_bgr(cv::Rect(input_eye_width, 0, input_eye_width, input_bgr.rows));

  cv::Mat left = letterboxToSize(left_raw, output_eye_width, height);
  cv::Mat right = letterboxToSize(right_raw, output_eye_width, height);
  if (left.empty() || right.empty()) {
    return cv::Mat();
  }

  cv::Mat sbs;
  cv::hconcat(left, right, sbs);
  return sbs;
}

cv::Mat bgrToBgraForEncoding(const cv::Mat &input_bgr, int width, int height) {
  cv::Mat bgr;
  if (input_bgr.cols == 1280 && input_bgr.rows == 480 &&
      width == 2560 && height == 720) {
    bgr = prepareStereoSbsFrame(input_bgr, width, height);
    static bool logged_stereo_letterbox = false;
    if (!logged_stereo_letterbox && !bgr.empty()) {
      std::cout << "Using stereo SBS letterbox: 1280x480 -> 2560x720 "
                << "(each eye 640x480 -> 1280x720 with preserved aspect)"
                << std::endl;
      logged_stereo_letterbox = true;
    }
  } else {
    bgr = input_bgr;
    if (bgr.cols != width || bgr.rows != height) {
      cv::resize(bgr, bgr, cv::Size(width, height), 0, 0, cv::INTER_LINEAR);
    }
  }

  if (bgr.empty()) {
    return cv::Mat();
  }

  cv::Mat bgra;
  cv::cvtColor(bgr, bgra, cv::COLOR_BGR2BGRA);
  return bgra;
}

bool recvLatestJpeg(void *socket, std::vector<uint8_t> &jpg,
                    int timeout_ms) {
  zmq_pollitem_t items[] = {{socket, 0, ZMQ_POLLIN, 0}};
  const int poll_rc = zmq_poll(items, 1, timeout_ms);
  if (poll_rc <= 0 || !(items[0].revents & ZMQ_POLLIN)) {
    return false;
  }

  zmq_msg_t msg;
  zmq_msg_init(&msg);
  int rc = zmq_msg_recv(&msg, socket, ZMQ_DONTWAIT);
  if (rc < 0) {
    zmq_msg_close(&msg);
    return false;
  }

  const uint8_t *data = static_cast<const uint8_t *>(zmq_msg_data(&msg));
  const size_t size = zmq_msg_size(&msg);
  jpg.assign(data, data + size);
  zmq_msg_close(&msg);
  return true;
}

void drainGstBus(GstElement *pipeline) {
  GstBus *bus = gst_element_get_bus(pipeline);
  while (true) {
    GstMessage *msg = gst_bus_pop_filtered(
        bus, static_cast<GstMessageType>(GST_MESSAGE_ERROR |
                                         GST_MESSAGE_WARNING |
                                         GST_MESSAGE_STATE_CHANGED |
                                         GST_MESSAGE_EOS));
    if (!msg) {
      break;
    }

    switch (GST_MESSAGE_TYPE(msg)) {
    case GST_MESSAGE_ERROR: {
      GError *err = nullptr;
      gchar *debug = nullptr;
      gst_message_parse_error(msg, &err, &debug);
      std::cerr << "GStreamer ERROR from "
                << GST_OBJECT_NAME(msg->src) << ": "
                << (err ? err->message : "unknown") << std::endl;
      if (debug) {
        std::cerr << "GStreamer debug: " << debug << std::endl;
      }
      if (err) {
        g_error_free(err);
      }
      if (debug) {
        g_free(debug);
      }
      break;
    }
    case GST_MESSAGE_WARNING: {
      GError *err = nullptr;
      gchar *debug = nullptr;
      gst_message_parse_warning(msg, &err, &debug);
      std::cerr << "GStreamer WARNING from "
                << GST_OBJECT_NAME(msg->src) << ": "
                << (err ? err->message : "unknown") << std::endl;
      if (err) {
        g_error_free(err);
      }
      if (debug) {
        g_free(debug);
      }
      break;
    }
    case GST_MESSAGE_EOS:
      std::cerr << "GStreamer EOS received" << std::endl;
      break;
    default:
      break;
    }
    gst_message_unref(msg);
  }
  gst_object_unref(bus);
}

bool checkEncoderPlugin(const std::string &encoder) {
  std::string element_name;
  if (encoder == "x264") {
    element_name = "x264enc";
  } else {
    element_name = "nvv4l2h264enc";
  }

  GstElementFactory *factory =
      gst_element_factory_find(element_name.c_str());
  if (!factory) {
    std::cerr << "Missing GStreamer H.264 encoder plugin: "
              << element_name << std::endl;
    if (encoder == "x264") {
      std::cerr << "Install it with: sudo apt install "
                << "gstreamer1.0-plugins-ugly" << std::endl;
    } else {
      std::cerr << "Install NVIDIA Jetson GStreamer plugins, or use "
                << "--encoder x264 after installing x264enc." << std::endl;
    }
    return false;
  }

  gst_object_unref(factory);
  return true;
}

bool checkRequiredGstPlugins(const std::string &encoder) {
  if (!checkEncoderPlugin(encoder)) {
    return false;
  }

  GstElementFactory *parser_factory = gst_element_factory_find("h264parse");
  if (!parser_factory) {
    std::cerr << "Missing GStreamer H.264 parser plugin: h264parse"
              << std::endl;
    std::cerr << "Install it with: sudo apt install "
              << "gstreamer1.0-plugins-bad" << std::endl;
    return false;
  }

  gst_object_unref(parser_factory);
  return true;
}

int main(int argc, char *argv[]) {
  printTimeMs("Start");
  gst_init(&argc, &argv);
  signal(SIGINT, handle_sigint);

  bool preview_enabled = false;
  bool listen_enabled = true;
  std::string listen_address = "0.0.0.0:13579";
  std::string teleimager_host = "127.0.0.1";
  int teleimager_port = 55555;
  int width = 2560;
  int height = 720;
  int fps = 60;
  int bitrate = 4000000;
  int debug_log_interval = 30;
  int zmq_timeout_ms = 100;
  std::string encoder = "x264";
  std::string save_input_path = "";

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--preview") {
      preview_enabled = true;
    } else if (arg == "--send") {
      listen_enabled = false;
      send_enabled = true;
      encoding_enabled = true;
    } else if (arg == "--listen" && i + 1 < argc) {
      listen_enabled = true;
      listen_address = argv[++i];
    } else if (arg == "--listen-autosend" && i + 1 < argc) {
      listen_enabled = true;
      listen_address = argv[++i];
      send_over_listen_socket = true;
      auto_send_on_client = true;
    } else if (arg == "--server" && i + 1 < argc) {
      send_to_server = argv[++i];
    } else if (arg == "--port" && i + 1 < argc) {
      send_to_port = std::stoi(argv[++i]);
    } else if (arg == "--teleimager-host" && i + 1 < argc) {
      teleimager_host = argv[++i];
    } else if (arg == "--teleimager-port" && i + 1 < argc) {
      teleimager_port = std::stoi(argv[++i]);
    } else if (arg == "--width" && i + 1 < argc) {
      width = std::stoi(argv[++i]);
    } else if (arg == "--height" && i + 1 < argc) {
      height = std::stoi(argv[++i]);
    } else if (arg == "--fps" && i + 1 < argc) {
      fps = std::stoi(argv[++i]);
    } else if (arg == "--bitrate" && i + 1 < argc) {
      bitrate = std::stoi(argv[++i]);
    } else if (arg == "--encoder" && i + 1 < argc) {
      encoder = argv[++i];
    } else if (arg == "--save-input" && i + 1 < argc) {
      save_input_path = argv[++i];
    } else if (arg == "--debug-log-interval" && i + 1 < argc) {
      debug_log_interval = std::stoi(argv[++i]);
    } else if (arg == "--zmq-timeout-ms" && i + 1 < argc) {
      zmq_timeout_ms = std::stoi(argv[++i]);
    } else if (arg == "--help") {
      std::cout << "Usage: " << argv[0] << " [options]\n"
                << "  --preview                 Enable local preview\n"
                << "  --listen IP:PORT          Listen to headset control commands\n"
                << "  --listen-autosend IP:PORT Listen and send video on the same connection\n"
                << "  --send                    Send immediately without waiting command\n"
                << "  --server IP               Headset/video server IP\n"
                << "  --port PORT               Headset/video server port\n"
                << "  --teleimager-host HOST    Teleimager ZMQ host\n"
                << "  --teleimager-port PORT    Teleimager head camera ZMQ port\n"
                << "  --width WIDTH             Encoded frame width, default 2560\n"
                << "  --height HEIGHT           Encoded frame height, default 720\n"
                << "  --fps FPS                 Encoded fps, default 60\n"
                << "  --bitrate BPS             H264 bitrate, default 4000000\n"
                << "  --encoder nvv4l2|x264     H264 encoder, default x264\n"
                << "  --save-input PATH         Save first decoded Teleimager frame\n"
                << "  --debug-log-interval N    Print frame status every N frames\n"
                << "  --zmq-timeout-ms N        ZMQ receive poll timeout, default 100\n";
      return 0;
    }
  }

  if (listen_enabled) {
    send_enabled = false;
    encoding_enabled = false;
    server_ptr.reset(new TCPServer(listen_address));
    server_ptr->setDataCallback(onDataCallback);
    server_ptr->setDisconnectCallback(onDisconnectCallback);
    server_ptr->start();
    std::cout << "TCPServer is listening on " << listen_address << std::endl;
  }

  if (send_enabled && !initialize_sender()) {
    return -1;
  }

  void *zmq_context = zmq_ctx_new();
  void *zmq_sub_socket = zmq_socket(zmq_context, ZMQ_SUB);
  int rcvhwm = 1;
  int linger = 0;
  zmq_setsockopt(zmq_sub_socket, ZMQ_RCVHWM, &rcvhwm, sizeof(rcvhwm));
  zmq_setsockopt(zmq_sub_socket, ZMQ_LINGER, &linger, sizeof(linger));
  std::string zmq_endpoint =
      "tcp://" + teleimager_host + ":" + std::to_string(teleimager_port);
  if (zmq_connect(zmq_sub_socket, zmq_endpoint.c_str()) != 0) {
    std::cerr << "Failed to connect Teleimager ZMQ endpoint: " << zmq_endpoint
              << std::endl;
    return -1;
  }
  zmq_setsockopt(zmq_sub_socket, ZMQ_SUBSCRIBE, "", 0);
  std::cout << "Connected to Teleimager ZMQ: " << zmq_endpoint << std::endl;

  const std::string pipeline_str =
      buildPipelineString(encoder, width, height, fps, bitrate, preview_enabled);
  std::cout << "GStreamer pipeline:\n" << pipeline_str << std::endl;

  if (!checkRequiredGstPlugins(encoder)) {
    return -1;
  }

  GError *error = nullptr;
  GstElement *pipeline = gst_parse_launch(pipeline_str.c_str(), &error);
  if (error) {
    std::cerr << "GStreamer parse warning/error: " << error->message
              << std::endl;
    g_clear_error(&error);
  }
  if (!pipeline) {
    std::cerr << "Failed to create pipeline" << std::endl;
    return -1;
  }

  GstElement *appsrc = gst_bin_get_by_name(GST_BIN(pipeline), "mysource");
  GstElement *appsink = gst_bin_get_by_name(GST_BIN(pipeline), "mysink");
  g_signal_connect(appsink, "new-sample", G_CALLBACK(on_new_sample), nullptr);
  GstStateChangeReturn state_ret =
      gst_element_set_state(pipeline, GST_STATE_PLAYING);
  if (state_ret == GST_STATE_CHANGE_FAILURE) {
    std::cerr << "Failed to set GStreamer pipeline to PLAYING" << std::endl;
    drainGstBus(pipeline);
    return -1;
  }
  std::cout << "GStreamer pipeline set to PLAYING" << std::endl;

  int frame_id = 0;
  int raw_frame_id = 0;
  int idle_poll_count = 0;
  bool saved_input_frame = false;
  std::vector<uint8_t> jpg;
  while (!stop_requested) {
    if (!recvLatestJpeg(zmq_sub_socket, jpg, zmq_timeout_ms) || jpg.empty()) {
      idle_poll_count++;
      if (debug_log_interval > 0 &&
          idle_poll_count % debug_log_interval == 0) {
        std::cout << "No Teleimager ZMQ frame received yet from "
                  << zmq_endpoint << std::endl;
      }
      continue;
    }
    idle_poll_count = 0;

    cv::Mat bgr = decodeJpegToBgr(jpg);
    if (bgr.empty()) {
      std::cerr << "Failed to decode Teleimager JPEG frame, bytes="
                << jpg.size() << std::endl;
      continue;
    }

    if (!save_input_path.empty() && !saved_input_frame) {
      if (cv::imwrite(save_input_path, bgr)) {
        std::cout << "Saved first Teleimager decoded frame to "
                  << save_input_path << " size=" << bgr.cols << "x"
                  << bgr.rows << std::endl;
      } else {
        std::cerr << "Failed to save Teleimager decoded frame to "
                  << save_input_path << std::endl;
      }
      saved_input_frame = true;
    }

    if (debug_log_interval > 0 &&
        raw_frame_id % debug_log_interval == 0) {
      std::cout << "Teleimager frame " << raw_frame_id
                << ": jpeg_bytes=" << jpg.size()
                << ", decoded=" << bgr.cols << "x" << bgr.rows
                << ", encode_target=" << width << "x" << height
                << ", encoding_enabled=" << encoding_enabled
                << ", send_enabled=" << send_enabled
                << ", pushed=" << pushed_frame_count
                << ", encoded=" << encoded_frame_count << std::endl;
      if (!encoding_enabled) {
        std::cout << "Waiting for StartRobotCameraStream from headset..."
                  << std::endl;
      }
    }
    raw_frame_id++;

    if (auto_send_on_client && server_ptr && server_ptr->isClientConnected()) {
      if (!encoding_enabled || !send_enabled) {
        std::cout << "Client connected, auto-starting Teleimager video send"
                  << std::endl;
      }
      encoding_enabled = true;
      send_enabled = true;
    }

    if (!encoding_enabled) {
      continue;
    }

    cv::Mat bgra = bgrToBgraForEncoding(bgr, width, height);
    if (bgra.empty()) {
      std::cerr << "Failed to convert Teleimager frame to BGRA" << std::endl;
      continue;
    }

    GstBuffer *buffer =
        gst_buffer_new_allocate(nullptr, bgra.total() * bgra.elemSize(), nullptr);
    GstMapInfo map;
    gst_buffer_map(buffer, &map, GST_MAP_WRITE);
    memcpy(map.data, bgra.data, bgra.total() * bgra.elemSize());
    gst_buffer_unmap(buffer, &map);

    GST_BUFFER_PTS(buffer) = gst_util_uint64_scale(frame_id, GST_SECOND, fps);
    GST_BUFFER_DURATION(buffer) = gst_util_uint64_scale(1, GST_SECOND, fps);
    GstFlowReturn push_ret =
        gst_app_src_push_buffer(GST_APP_SRC(appsrc), buffer);
    pushed_frame_count++;
    if (push_ret != GST_FLOW_OK) {
      std::cerr << "gst_app_src_push_buffer failed: "
                << gst_flow_get_name(push_ret) << std::endl;
    }
    if (debug_log_interval > 0 &&
        pushed_frame_count % debug_log_interval == 1) {
      std::cout << "Pushed raw frame " << pushed_frame_count
                << " to GStreamer, flow=" << gst_flow_get_name(push_ret)
                << std::endl;
    }
    drainGstBus(pipeline);
    frame_id++;
  }

  if (sender_ptr) {
    sender_ptr->disconnect();
  }
  if (server_ptr) {
    server_ptr->stop();
  }

  gst_app_src_end_of_stream(GST_APP_SRC(appsrc));
  gst_element_set_state(pipeline, GST_STATE_NULL);
  gst_object_unref(appsrc);
  gst_object_unref(appsink);
  gst_object_unref(pipeline);
  zmq_close(zmq_sub_socket);
  zmq_ctx_term(zmq_context);

  return 0;
}
