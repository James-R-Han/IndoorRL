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
 * \file stereo_window_optimization_module.hpp
 * \brief StereoWindowOptimizationModule class definition
 *
 * \author Autonomous Space Robotics Lab (ASRL)
 */
#pragma once

// LGMath and Steam
#include <lgmath.hpp>
#include <steam.hpp>

#include <vtr_tactic/modules/base_module.hpp>
#include <vtr_vision/cache.hpp>
#include <vtr_vision/modules/optimization/steam_module.hpp>

namespace vtr {
namespace vision {

/** \brief A module that runs STEAM on multiple graph vertices. */
class StereoWindowOptimizationModule : public SteamModule {
 public:
  /** \brief Static module identifier. */
  static constexpr auto static_name = "stereo_window_optimization";

  /** \brief Collection of config parameters */
  struct Config : SteamModule::Config {
    bool depth_prior_enable;
    double depth_prior_weight;
  };

  StereoWindowOptimizationModule(const std::string &name = static_name)
      : SteamModule(name) {}

  void configFromROS(const rclcpp::Node::SharedPtr &node,
                     const std::string param_prefix) override;

 protected:
  /** \brief Update the graph with optimized transforms */
  virtual void updateGraphImpl(tactic::QueryCache &,
                               const tactic::Graph::Ptr &graph,
                               tactic::VertexId);

  /** \brief Given two frames, builds a sensor specific optimization problem. */
  virtual std::shared_ptr<steam::OptimizationProblem>
  generateOptimizationProblem(CameraQueryCache &qdata,
                              const tactic::Graph::ConstPtr &graph) override;

  void updateCaches(CameraQueryCache &qdata) override;

 private:
#if false
  /**
   * \brief samples and saves the optimized trajectory and stores it in the
   * latest vertex.
   */
  void saveTrajectory(CameraQueryCache &qdata,
                      const std::shared_ptr<Graph> &graph);
#endif
  /**
   * \brief Initializes the problem based on an initial condition.
   * The initial guess at the transformation between the query frame and the map
   * frame.
   */
  void resetProblem();

  /**
   * \brief Adds a depth cost associated with this landmark to the depth cost
   * terms.
   * \param landmark The landmark in question.
   */
  void addDepthCost(steam::se3::LandmarkStateVar::Ptr landmark);

  /**
   * \brief Verifies the input data being used in the optimization problem,
   * namely, the inlier matches and initial estimate.
   * \param qdata The query data.
   */
  bool verifyInputData(CameraQueryCache &qdata) override;

  /**
   * \brief Verifies the output data generated byt the optimization problem
   * \param qdata The query data.
   */
  bool verifyOutputData(CameraQueryCache &qdata) override;

  /**
   * \brief performs sanity checks on the landmark
   * \param point The landmark.
   * \param qdata The query data.*
   * \return true if the landmark meets all checks, false otherwise.
   */
  bool isLandmarkValid(const Eigen::Vector3d &point, CameraQueryCache &qdata);

  /** \brief the cost terms associated with landmark observations. */
  steam::ParallelizedCostTermCollection::Ptr cost_terms_;

  /** \brief The cost terms associated with landmark depth. */
  steam::ParallelizedCostTermCollection::Ptr depth_cost_terms_;

  /** \brief The loss function used for the depth cost. */
  steam::LossFunctionBase::Ptr sharedDepthLossFunc_;

  /** \brief the loss function assicated with observation cost. */
  steam::LossFunctionBase::Ptr sharedLossFunc_;

  /** \brief The steam problem. */
  std::shared_ptr<steam::OptimizationProblem> problem_;

  /** \brief Module configuration. */
  std::shared_ptr<Config> window_config_;
};

}  // namespace vision
}  // namespace vtr
