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
 * \file localization_icp_module_v2.hpp
 * \brief LocalizationICPModuleV2 class definition
 *
 * \author Yuchen Wu, Autonomous Space Robotics Lab (ASRL)
 */
#pragma once

#include <random>

#include <steam.hpp>

#include "vtr_common/timing/stopwatch.hpp"
#include "vtr_lidar/cache.hpp"
#include "vtr_tactic/modules/base_module.hpp"
#include "vtr_tactic/task_queue.hpp"

namespace vtr {

namespace lidar {

/** \brief ICP for localization. */
class LocalizationICPModuleV2 : public tactic::BaseModule {
 public:
  /** \brief Static module identifier. */
  static constexpr auto static_name = "lidar.localization_icp_v2";

  /** \brief Config parameters. */
  struct Config : public tactic::BaseModule::Config,
                  public steam::VanillaGaussNewtonSolver::Params {
    using Ptr = std::shared_ptr<Config>;
    using ConstPtr = std::shared_ptr<const Config>;

    /// Success criteria
    float min_matched_ratio = 0.4;

    /// Prior terms
    bool use_pose_prior = false;

    /// ICP parameters
    // number of threads for nearest neighbor search
    int num_threads = 8;
    // initial alignment config
    size_t first_num_steps = 3;
    size_t initial_max_iter = 100;
    size_t initial_num_samples = 1000;
    float initial_max_pairing_dist = 2.0;
    float initial_max_planar_dist = 0.3;
    // refined stage
    size_t refined_max_iter = 10;  // we use a fixed number of iters for now
    size_t refined_num_samples = 5000;
    float refined_max_pairing_dist = 2.0;
    float refined_max_planar_dist = 0.1;
    // error calculation
    float averaging_num_steps = 5;
    float trans_diff_thresh = 0.01;              // threshold on variation of T
    float rot_diff_thresh = 0.1 * M_PI / 180.0;  // threshold on variation of R

    static ConstPtr fromROS(const rclcpp::Node::SharedPtr &node,
                            const std::string &param_prefix);
  };

  LocalizationICPModuleV2(
      const Config::ConstPtr &config,
      const std::shared_ptr<tactic::ModuleFactoryV2> &module_factory = nullptr,
      const std::string &name = static_name)
      : tactic::BaseModule{module_factory, name}, config_(config) {}

 private:
  void runImpl(tactic::QueryCache &qdata, const tactic::Graph::Ptr &graph,
               const tactic::TaskExecutor::Ptr &executor) override;

  void addPosePrior(
      const tactic::EdgeTransform &T_r_m,
      const steam::se3::TransformEvaluator::Ptr &T_r_m_eval,
      const steam::ParallelizedCostTermCollection::Ptr &prior_cost_terms);

  Config::ConstPtr config_;

  VTR_REGISTER_MODULE_DEC_TYPE(LocalizationICPModuleV2);
};

}  // namespace lidar
}  // namespace vtr