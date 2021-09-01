// Copyright 2021, Autonomous Space Robotics Lab (ASRL)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/**
 * \file data_stream_writer.hpp
 * \brief
 * \details
 *
 * \author Autonomous Space Robotics Lab (ASRL)
 */
#pragma once

#include <any>
#include <utility>
#include <vtr_messages/msg/rig_calibration.hpp>
#include <vtr_storage/data_stream_base.hpp>
#include <vtr_storage/message.hpp>
#include <vtr_storage/sequential_append_writer.hpp>

namespace vtr {
namespace storage {

class DataStreamWriterBase : public DataStreamBase {
 public:
  DataStreamWriterBase(const std::string &data_directory_string,
                       const std::string &stream_name = "", bool append = false)
      : DataStreamBase(data_directory_string, stream_name), append_(append) {}
  virtual ~DataStreamWriterBase(){};

  virtual void open() = 0;
  virtual void close() = 0;
  virtual int32_t write(const VTRMessage &anytype_message) = 0;

 protected:
  virtual rosbag2_storage::TopicMetadata createTopicMetadata() = 0;

  bool append_;
};

template <typename MessageType>
class DataStreamWriter : public DataStreamWriterBase {
 public:
  DataStreamWriter(const std::string &data_directory_string,
                   const std::string &stream_name = "", bool append = false);
  ~DataStreamWriter();

  void open() override;
  void close() override;

  // returns the id of the inserted message
  int32_t write(const VTRMessage &vtr_message) override;

 protected:
  rosbag2_storage::TopicMetadata createTopicMetadata() override;

  rclcpp::Serialization<MessageType> serialization_;
  rosbag2_storage::TopicMetadata tm_;
  std::shared_ptr<SequentialAppendWriter> writer_;
};

class DataStreamWriterCalibration
    : public DataStreamWriter<vtr_messages::msg::RigCalibration> {
 public:
  DataStreamWriterCalibration(const std::string &data_directory_string)
      : DataStreamWriter(data_directory_string, CALIBRATION_FOLDER, false) {}
};

}  // namespace storage
}  // namespace vtr

#include "vtr_storage/data_stream_writer.inl"
