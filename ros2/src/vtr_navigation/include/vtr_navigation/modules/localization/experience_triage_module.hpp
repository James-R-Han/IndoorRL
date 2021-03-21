#pragma once

#include <vtr_navigation/modules/base_module.hpp>
#include <vtr_messages/msg/exp_recog_status.hpp>

namespace std {
/// Display the experience recognition status message in a readable way
std::ostream &operator<<(std::ostream &os, const vtr_messages::msg::ExpRecogStatus &msg);
}

namespace vtr {
namespace navigation {

/// Given a subgraph, return the run ids present
RunIdSet
getRunIds(const pose_graph::RCGraphBase &subgraph); ///<

/// Given a set of run ids, return those that are privileged (manual) runs
RunIdSet
privelegedRuns(const pose_graph::RCGraphBase &graph, ///< The graph (has manual info)
               RunIdSet rids); ///< The set of run ids to check

/// Given a subgraph, keep only vertices from runs in the mask
pose_graph::RCGraphBase::Ptr
maskSubgraph(const pose_graph::RCGraphBase::Ptr &graph,
             const RunIdSet &mask);

/// Fills up an experience recommendation set with n recommendations
RunIdSet /// @returns the newly inserted runs
fillRecommends(RunIdSet *recommends, ///< (optional) recommendation to fill
               const ScoredRids &distance_rids, ///< run similarity scores
               unsigned n); ///< the max number of runs in the recommendation

/// Mask the localization subgraph based on upstream experience (run) recommendations
class ExperienceTriageModule : public BaseModule {
 public:
  template<class T> using Ptr = std::shared_ptr<T>;

  /// Module name
  static constexpr auto type_str_ = "experience_triage";

  /// Module Configuration
  struct Config {
    bool in_the_loop = true;
    bool verbose = false;
    bool always_privileged = true;
  };

  /// Masks the localization map by the experience mask generated from upstream recommenders
  virtual void run(
      QueryCache &qdata, ///< Information about the live image
      MapCache &mdata, ///< Information about the map
      const std::shared_ptr<const Graph> &graph); ///< The spatio-temporal pose graph

  /// Saves the status message to the pose graph
  virtual void updateGraph(
      QueryCache &qdata, ///< Information about the live image
      MapCache &mdata, ///< Information about the map
      const std::shared_ptr<Graph> &graph, ///< The pose grpah
      VertexId live_id);  ///< The live vertex id

  /// Sets the module configuration.
  void setConfig(const Config &config) { ///< The new config
    config_ = config;
  }

 private:

  /// The status message to save to the graph
  vtr_messages::msg::ExpRecogStatus status_msg_;

  /// The module configuration
  Config config_;
};

} // navigation
} // asrl