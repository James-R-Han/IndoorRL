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
 * \file base_path_planner.hpp
 * \author Yuchen Wu, Autonomous Space Robotics Lab (ASRL)
 */
#pragma once

#include <chrono>
#include <condition_variable>
#include <mutex>
#include <thread>

#include "vtr_logging/logging.hpp"
#include "vtr_path_planning/path_planner_interface.hpp"
#include "vtr_tactic/cache.hpp"

#include "geometry_msgs/msg/twist.hpp"

namespace vtr {
namespace path_planning {

class PathPlannerCallbackInterface {
 public:
  PTR_TYPEDEFS(PathPlannerCallbackInterface);

  using Command = geometry_msgs::msg::Twist;

  virtual ~PathPlannerCallbackInterface() = default;

  virtual tactic::Timestamp getCurrentTime() const {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::system_clock::now().time_since_epoch())
        .count();
  }

  /** \brief Callback when a new route is set */
  virtual void stateChanged(const bool& /* running */) {}

  /**
   * \brief Callback when the route is finished
   * \note command is not a class member of BasePathPlanner, so this function is
   * called without BasePathPlanner locked
   */
  virtual void commandReceived(const Command& /* command */) {}
};

class BasePathPlanner : public PathPlannerInterface {
 public:
  PTR_TYPEDEFS(BasePathPlanner);

  using Mutex = std::mutex;
  using CondVar = std::condition_variable;
  using LockGuard = std::lock_guard<Mutex>;
  using UniqueLock = std::unique_lock<Mutex>;

  using RobotState = tactic::OutputCache;
  using Command = geometry_msgs::msg::Twist;
  using Callback = PathPlannerCallbackInterface;

  struct Config {
    PTR_TYPEDEFS(Config);

    unsigned int control_period = 0;

    static Ptr fromROS(const rclcpp::Node::SharedPtr& node,
                       const std::string& prefix = "path_planning");
  };

  BasePathPlanner(const Config::Ptr& config, const RobotState::Ptr& robot_state,
                  const Callback::Ptr& callback = std::make_shared<Callback>());
  ~BasePathPlanner() override;

  /// for state machine use
 public:
  /** \brief Call when a new route is set */
  void initializeRoute() override;
  /** \brief Sets whether control command should be computed */
  void setRunning(const bool running) override;

 private:
  /** \brief Called when a new route is set */
  virtual void initializeRoute(RobotState& robot_state) = 0;
  /** \brief Subclass override this method to compute a control command */
  virtual Command computeCommand(RobotState& robot_state) = 0;

 protected:
  /** \brief Subclass use this function to get current time */
  tactic::Timestamp now() const { return callback_->getCurrentTime(); }
  /** \brief Derived class must call this upon destruction */
  void stop();

 private:
  void process();

 private:
  const Config::ConstPtr config_;
  /** \brief shared memory that stores the current robot state */
  const RobotState::Ptr robot_state_;
  /** \brief callback functions on control finished */
  const Callback::Ptr callback_;

 protected:
  /** \brief Protects all class members */
  mutable Mutex mutex_;
  /** \brief wait until stop or controller state has changed */
  mutable CondVar cv_terminate_or_state_changed_;
  mutable CondVar cv_thread_finish_;

 private:
  /** \brief Whether the processing thread is computing command */
  bool running_ = false;

  /** \brief signal the process thread to stop */
  bool terminate_ = false;
  /** \brief used to wait until all threads finish */
  size_t thread_count_ = 0;
  /** \brief the event processing thread */
  std::thread process_thread_;
};

}  // namespace path_planning
}  // namespace vtr
