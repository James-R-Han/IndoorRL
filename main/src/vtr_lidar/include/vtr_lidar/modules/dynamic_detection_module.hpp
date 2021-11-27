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
 * \file dynamic_detection_module.hpp
 * \brief DynamicDetectionModule class definition
 *
 * \author Yuchen Wu, Autonomous Space Robotics Lab (ASRL)
 */
#pragma once

#include "vtr_lidar/cache.hpp"
#include "vtr_lidar/modules/intra_exp_merging_module.hpp"
#include "vtr_tactic/modules/base_module.hpp"
#include "vtr_tactic/modules/module_factory.hpp"
#include "vtr_tactic/task_queue.hpp"

// visualization
#include "sensor_msgs/msg/point_cloud2.hpp"

namespace vtr {
namespace lidar {

/**
 * \brief Uses ray-tracing to detect short-term dynamic objects.
 * Asynchronous. Optional.
 */
class DynamicDetectionModule : public tactic::BaseModule {
 public:
  using Ptr = std::shared_ptr<DynamicDetectionModule>;
  using WeakPtr = std::weak_ptr<DynamicDetectionModule>;
  using PointCloudMsg = sensor_msgs::msg::PointCloud2;

  static constexpr auto static_name = "lidar.dynamic_detection";

  /** \brief Collection of config parameters */
  struct Config {
    int depth = 0;

    // dependencies
    std::string intra_exp_merging = IntraExpMergingModule::static_name;

    float horizontal_resolution = 0.001;
    float vertical_resolution = 0.001;
    int max_num_observations = 20;
    int min_num_observations = 0;
    float dynamic_threshold = 0.5;

    bool visualize = false;
  };

  DynamicDetectionModule(const std::string &name = static_name)
      : BaseModule{name}, config_(std::make_shared<Config>()) {}

  void configFromROS(const rclcpp::Node::SharedPtr &node,
                     const std::string param_prefix) override;

  std::shared_ptr<const Config> config() const { return config_; }

  rclcpp::Publisher<PointCloudMsg>::SharedPtr &oldMapPublisher() {
    return old_map_pub_;
  }
  rclcpp::Publisher<PointCloudMsg>::SharedPtr &newMapPublisher() {
    return new_map_pub_;
  }
  rclcpp::Publisher<PointCloudMsg>::SharedPtr &scanPublisher() {
    return scan_pub_;
  }

 private:
  void runImpl(tactic::QueryCache &qdata, const tactic::Graph::Ptr &graph,
               const tactic::TaskExecutor::Ptr &executor) override;

  void runAsyncImpl(tactic::QueryCache &qdata, const tactic::Graph::Ptr &graph,
                    const tactic::TaskExecutor::Ptr &executor,
                    const tactic::Task::Priority &priority,
                    const tactic::Task::DepId &dep_id) override;

  /** \brief Module configuration. */
  std::shared_ptr<Config> config_;  /// \todo no need to be a shared pointer.

  /** \brief mutex to make publisher thread safe */
  std::mutex mutex_;

  /** \brief for visualization only */
  bool publisher_initialized_ = false;
  rclcpp::Publisher<PointCloudMsg>::SharedPtr old_map_pub_;
  rclcpp::Publisher<PointCloudMsg>::SharedPtr new_map_pub_;
  rclcpp::Publisher<PointCloudMsg>::SharedPtr scan_pub_;
};

}  // namespace lidar
}  // namespace vtr