#pragma once

#include <rclcpp/rclcpp.hpp>

#include <vtr_messages/msg/rig_images.hpp>
#include <vtr_storage/data_stream_reader.hpp>
#include <vtr_storage/data_stream_writer.hpp>

using RigImages = vtr_messages::msg::RigImages;
using RigCalibration = vtr_messages::msg::RigCalibration;

class Xb3Replay : public rclcpp::Node {
 public:
  Xb3Replay(const std::string &data_dir, const std::string &stream_name,
            const std::string &topic, int qos = 10);

  vtr::storage::DataStreamReader<RigImages, RigCalibration> reader_;

  rclcpp::Publisher<RigImages>::SharedPtr publisher_;

};