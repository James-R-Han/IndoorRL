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
 * \file data_stream_base.cpp
 * \brief
 * \details
 *
 * \author Autonomous Space Robotics Lab (ASRL)
 */
#include <vtr_storage/data_stream_base.hpp>

namespace vtr {
namespace storage {

DataStreamBase::DataStreamBase(const std::string &data_directory_string,
                               const std::string &stream_name)
    : base_directory_(rcpputils::fs::path(data_directory_string)),
      data_directory_(rcpputils::fs::path(data_directory_string)),
      stream_name_(stream_name),
      opened_(false) {
  if (stream_name != "") {
    data_directory_ /= stream_name;
  }
  storage_options_.uri = data_directory_.string();
  storage_options_.storage_id = "sqlite3";
  storage_options_.max_bagfile_size = 0;
  storage_options_.max_cache_size = 0;

  converter_options_.input_serialization_format = "cdr";
  converter_options_.output_serialization_format = "cdr";
}

DataStreamBase::~DataStreamBase() {}
}  // namespace storage
}  // namespace vtr