#pragma once
#if 0
#include <map>

#include <vtr_pose_graph/index/rc_graph.hpp>
#include <vtr_pose_graph/relaxation/privileged_frame.hpp>
//#include <asrl/messages/GPSMeasurement.pb.h>
#include <babelfish_robochunk_sensor_msgs/NavSatFix.pb.h>
#endif

namespace vtr {
namespace pose_graph {
#if 0
class GpsAlignedFrame : public PrivilegedFrame<RCGraph> {
 public:
  typedef RCGraph::VertexPtr VertexPtr;
  typedef RCGraph::EdgePtr EdgePtr;
  typedef RCGraph::VertexIdType VertexIdType;
  typedef RCGraph::TransformType TransformType;
  typedef RCGraph::OrderedIter IterType;

  typedef PrivilegedFrame<RCGraph> Base;
  //  typedef asrl::sensor_msgs::GPSMeasurement GpsMsgType;
  typedef babelfish::robochunk::sensor_msgs::NavSatFix GpsMsgType;

  static constexpr char gpsStream[] = "/DGPS/fix";

  DEFAULT_COPY_MOVE(GpsAlignedFrame)

  /////////////////////////////////////////////////////////////////////////////
  /// @brief Constructor
  /////////////////////////////////////////////////////////////////////////////
  GpsAlignedFrame(const RCGraphBase::Ptr& graph, IterType begin,
                  bool lazy = true);

  /////////////////////////////////////////////////////////////////////////////
  /// @brief Get the computed UTM zone
  /////////////////////////////////////////////////////////////////////////////
  inline uint32_t zone() const { return utmZone_; }

  /////////////////////////////////////////////////////////////////////////////
  /// @brief Get the computed rotation
  /////////////////////////////////////////////////////////////////////////////
  inline const Eigen::Matrix3d& C() const { return C_; }

  /////////////////////////////////////////////////////////////////////////////
  /// @brief Get the computed offset
  /////////////////////////////////////////////////////////////////////////////
  inline const Eigen::Vector3d& r() const { return r_; }

 private:
  uint32_t utmZone_;

  Eigen::Matrix3d C_;

  Eigen::Vector3d r_;
};
#endif

}  // namespace pose_graph
}  // namespace vtr

#include <vtr_pose_graph/relaxation/privileged_frame.inl>