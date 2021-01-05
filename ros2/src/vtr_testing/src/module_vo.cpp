#include <filesystem>
#include <iostream>

#include "rclcpp/rclcpp.hpp"

#include <vtr_common/utils/filesystem.hpp>
#include <vtr_logging/logging_init.hpp>
#include <vtr_testing/module_vo.hpp>

#if 0
#include <tf/transform_listener.h>

#include <robochunk_msgs/MessageBase.pb.h>
#include <robochunk_msgs/XB3CalibrationRequest.pb.h>
#include <robochunk_msgs/XB3CalibrationResponse.pb.h>
#include <robochunk/base/DataStream.hpp>
#include <robochunk/util/fileUtils.hpp>

#include <vtr/vision/messages/bridge.h>
#include <asrl/common/timing/SimpleTimer.hpp>
#endif

using namespace vtr::common::utils;
using namespace vtr::logging;
using RigImages = vtr_messages::msg::RigImages;
using RigCalibration = vtr_messages::msg::RigCalibration;

int main(int argc, char** argv) {
  // easylogging++ configuration
  configureLogging();
  
  LOG(INFO) << "Starting Module VO, beep beep beep";

  /// // enable parallelisation
  /// Eigen::initParallel();  // This is no longer needed in Eigen3?

  rclcpp::init(argc, argv);
  auto node = rclcpp::Node::make_shared("module_vo");
  auto data_dir_str =
      node->declare_parameter<std::string>("input_data_dir", "");
  auto results_dir_str =
      node->declare_parameter<std::string>("results_dir", "");
  auto sim_run_str = node->declare_parameter<std::string>("sim_run", "");
  auto stream_name = node->declare_parameter<std::string>("stream_name", "");

  fs::path data_dir{expand_user(data_dir_str)};
  fs::path results_dir{expand_user(results_dir_str)};
  fs::path sim_run{expand_user(sim_run_str)};

  auto start_index = node->declare_parameter<int>("start_index", 1);
  auto stop_index = node->declare_parameter<int>("stop_index", 20000);

  ModuleVO vo(node, results_dir);

  vtr::storage::DataStreamReader<RigImages, RigCalibration> stereo_stream(
      data_dir.string(), stream_name);
  vtr::vision::RigCalibration rig_calibration;

  try {
    auto calibration_msg =
        stereo_stream.fetchCalibration()->get<RigCalibration>();
    rig_calibration = vtr::messages::copyCalibration(calibration_msg);
  } catch (vtr::storage::NoBagExistsException& e) {
    LOG(ERROR) << "No calibration message recorded! URI: "
               << e.get_directory().string();
    return -1;
  }

  vo.setCalibration(
      std::make_shared<vtr::vision::RigCalibration>(rig_calibration));

  bool seek_success =
      stereo_stream.seekByIndex(static_cast<int32_t>(start_index));
  if (!seek_success) {
    LOG(ERROR) << "Seek failed!";
    return 0;
  }

  std::shared_ptr<vtr::storage::VTRMessage> storage_msg;
  int idx = 0;
  while (idx + start_index < stop_index && rclcpp::ok()) {
    storage_msg = stereo_stream.readNextFromSeek();
    if (!storage_msg) {
      LOG(ERROR) << "Storage msg is nullptr!";
      break;
    }
    auto rig_images = storage_msg->template get<RigImages>();
    // \todo current datasets didn't fill vtr_header so need this line
    rig_images.vtr_header.sensor_time_stamp.nanoseconds_since_epoch = rig_images.channels[0].cameras[0].stamp.nanoseconds_since_epoch;
    auto timestamp = rig_images.vtr_header.sensor_time_stamp;
    LOG(INFO) << "\nProcessing image: " << idx;
    vo.processImageData(std::make_shared<RigImages>(rig_images), timestamp);
    idx++;
  }
  LOG(INFO) << "Time to exit!";

/// \todo yuchen old code as reference
#if 0
  robochunk::base::ChunkStream stereo_stream(data_dir / sim_run, stream_name);
  robochunk::msgs::RobochunkMessage calibration_msg;

  LOG(INFO) << "Fetching calibration...";
  // Check out the calibration
  if (stereo_stream.fetchCalibration(calibration_msg) == true) {
    LOG(INFO) << "Calibration fetched...";
    std::shared_ptr<vtr::vision::RigCalibration> rig_calib = nullptr;

    LOG(INFO) << "Trying to extract a sensor_msgs::RigCalibration...";
    auto rig_calibration =
        calibration_msg
            .extractSharedPayload<robochunk::sensor_msgs::RigCalibration>();
    if (rig_calibration == nullptr) {
      LOG(WARNING)
          << "Trying to extract a sensor_msgs::RigCalibration failed, so I'm "
             "going to try to extract a sensor_msgs::XB3CalibrationResponse...";
      auto xb3_calibration = calibration_msg.extractSharedPayload<
          robochunk::sensor_msgs::XB3CalibrationResponse>();
      if (xb3_calibration == nullptr) {
        LOG(ERROR) << "Trying to extract a sensor_msgs::XB3CalibrationResponse "
                      "failed. Calibration extraction failed!!";
      } else {
        LOG(INFO)
            << "Successfully extracted a sensor_msgs::XB3CalibrationResponse.";
        rig_calib = std::make_shared<vtr::vision::RigCalibration>(
            vtr::messages::copyCalibration(*xb3_calibration));
      }
    } else {
      LOG(INFO) << "Successfully extracted a sensor_msgs::RigCalibration.";
      rig_calib = std::make_shared<vtr::vision::RigCalibration>(
          vtr::messages::copyCalibration(*rig_calibration));
    }

    if (rig_calib != nullptr) {
      LOG(INFO) << "Received camera calibration!";
      vo.setCalibration(rig_calib);
    } else {
      LOG(ERROR) << "ERROR: intrinsic params is not the correct type: (actual: "
                 << calibration_msg.header().type_name().c_str() << ")";
      return -1;
    }
  } else {
    LOG(ERROR) << "ERROR: Could not read calibration message!";
  }

  // Seek to an absolute index
  stereo_stream.seek(static_cast<uint32_t>(start_index));

  // Get the first message
  bool continue_stream = true;
  robochunk::msgs::RobochunkMessage data_msg;
  continue_stream &= stereo_stream.next(data_msg);
  int idx = 0;
  asrl::common::timing::SimpleTimer timer;  // unused
  while (continue_stream == true && idx + start_index < stop_index &&
         ros::ok()) {
    auto rig_images =
        data_msg.extractSharedPayload<robochunk::sensor_msgs::RigImages>();
    if (rig_images != nullptr) {
      LOG(INFO) << "Processing the " << idx << "th image";
      vo.processImageData(rig_images, data_msg.header().sensor_time_stamp());
    } else
      LOG(ERROR) << "Data is nullptr!";
    data_msg.Clear();
    continue_stream &= stereo_stream.next(data_msg);
    idx++;
  }
  LOG(INFO) << "Time to exit!";
#endif
}