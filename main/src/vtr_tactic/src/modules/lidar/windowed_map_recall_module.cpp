#include <vtr_tactic/modules/lidar/windowed_map_recall_module.hpp>

namespace {
void retrievePointMap(const PointMapMsg::SharedPtr &map_msg,
                      std::vector<PointXYZ> &points,
                      std::vector<PointXYZ> &normals,
                      std::vector<float> &scores,
                      std::vector<std::pair<int, int>> &movabilities) {
  auto N = map_msg->points.size();
  points.reserve(N);
  normals.reserve(N);
  scores.reserve(N);
  movabilities.reserve(N);

  for (unsigned i = 0; i < N; i++) {
    const auto &point = map_msg->points[i];
    const auto &normal = map_msg->normals[i];
    const auto &score = map_msg->scores[i];
    const auto &mb = map_msg->movabilities[i];
    // Add all points to the vector container
    points.push_back(PointXYZ(point.x, point.y, point.z));
    normals.push_back(PointXYZ(normal.x, normal.y, normal.z));
    scores.push_back(score);
    movabilities.push_back(std::pair<int, int>(mb.dynamic_obs, mb.total_obs));
  }
}

void migratePoints(const lgmath::se3::TransformationWithCovariance &T,
                   std::vector<PointXYZ> &points,
                   std::vector<PointXYZ> &normals) {
  /// Transform subsampled points into the map frame
  const auto T_mat = T.matrix();
  Eigen::Map<Eigen::Matrix<float, 3, Eigen::Dynamic>> pts_mat(
      (float *)points.data(), 3, points.size());
  Eigen::Map<Eigen::Matrix<float, 3, Eigen::Dynamic>> norms_mat(
      (float *)normals.data(), 3, normals.size());
  Eigen::Matrix3f R_tot = (T_mat.block<3, 3>(0, 0)).cast<float>();
  Eigen::Vector3f T_tot = (T_mat.block<3, 1>(0, 3)).cast<float>();
  pts_mat = (R_tot * pts_mat).colwise() + T_tot;
  norms_mat = R_tot * norms_mat;
}

}  // namespace

namespace vtr {
namespace tactic {
namespace lidar {

void WindowedMapRecallModule::configFromROS(const rclcpp::Node::SharedPtr &node,
                                            const std::string param_prefix) {
  config_ = std::make_shared<Config>();
  // clang-format off
  config_->map_voxel_size = node->declare_parameter<float>(param_prefix + ".map_voxel_size", config_->map_voxel_size);
  config_->depth = node->declare_parameter<int>(param_prefix + ".depth", config_->depth);
  config_->visualize = node->declare_parameter<bool>(param_prefix + ".visualize", config_->visualize);
  // clang-format on
}

void WindowedMapRecallModule::runImpl(QueryCache &qdata, MapCache &,
                                      const Graph::ConstPtr &graph) {
  if (config_->visualize && !publisher_initialized_) {
    // clang-format off
    map_pub_ = qdata.node->create_publisher<PointCloudMsg>("loc_map_pts_obs", 5);
    movability_map_pub_ = qdata.node->create_publisher<PointCloudMsg>("loc_map_pts_mvblty", 5);
    // clang-format on
    publisher_initialized_ = true;
  }

  // input
  auto &map_id = *qdata.map_id;

  // load vertex data
  auto run = graph->run(map_id.majorId());
  run->registerVertexStream<PointMapMsg>("pointmap", true,
                                         pose_graph::RegisterMode::Existing);

  auto map = std::make_shared<vtr::lidar::IncrementalPointMap>(
      config_->map_voxel_size);
  if (config_->depth == 0) {
    /// Recall a single map
    auto vertex = graph->at(map_id);
    vertex->load("pointmap");  /// \todo shouldn't this be in retrieve?
    const auto &map_msg = vertex->retrieveKeyframeData<PointMapMsg>("pointmap");
    std::vector<PointXYZ> points;
    std::vector<PointXYZ> normals;
    std::vector<float> scores;
    std::vector<std::pair<int, int>> movabilities;
    retrievePointMap(map_msg, points, normals, scores, movabilities);
    map->update(points, normals, scores, movabilities);
  } else {
    /// Recall multiple map
    // Iterate on the temporal edges to get the window.
    graph->lock();
    PrivilegedEvaluator::Ptr evaluator(new PrivilegedEvaluator());
    evaluator->setGraph((void *)graph.get());
    std::vector<VertexId> vertices;
    auto itr = graph->beginBfs(map_id, config_->depth, evaluator);
    for (; itr != graph->end(); ++itr) {
      const auto current_vertex = itr->v();
      // add the current, privileged vertex.
      vertices.push_back(current_vertex->id());
    }
    auto sub_graph = graph->getSubgraph(vertices);
    graph->unlock();

    // cache all the transforms so we only calculate them once
    pose_graph::PoseCache<pose_graph::RCGraph> pose_cache(graph, map_id);

    // construct the map
    for (auto vid : sub_graph->subgraph().getNodeIds()) {
      LOG(INFO) << "[lidar.windowed_recall] Looking at vertex: " << vid;
      // get transformation
      auto T_root_curr = pose_cache.T_root_query(vid);
      // migrate submaps
      auto vertex = graph->at(vid);
      vertex->load("pointmap");  /// \todo should  be in retrieveKeyframeData?
      const auto &map_msg =
          vertex->retrieveKeyframeData<PointMapMsg>("pointmap");
      std::vector<PointXYZ> points;
      std::vector<PointXYZ> normals;
      std::vector<float> scores;
      std::vector<std::pair<int, int>> movabilities;
      retrievePointMap(map_msg, points, normals, scores, movabilities);
      migratePoints(T_root_curr, points, normals);
      map->update(points, normals, scores, movabilities);
    }
  }

  if (config_->visualize) {
    {
      // publish map and number of observations of each point
      auto pc2_msg = std::make_shared<PointCloudMsg>();
      pcl::PointCloud<pcl::PointXYZI> cloud;
      cloud.reserve(map->cloud.pts.size());

      auto pcitr = map->cloud.pts.begin();
      auto ititr = map->movabilities.begin();
      for (; pcitr != map->cloud.pts.end(); pcitr++, ititr++) {
        pcl::PointXYZI pt;
        pt.x = pcitr->x;
        pt.y = pcitr->y;
        pt.z = pcitr->z;
        pt.intensity = ititr->second;
        cloud.points.push_back(pt);
      }

      pcl::toROSMsg(cloud, *pc2_msg);
      pc2_msg->header.frame_id = "localization keyframe";
      pc2_msg->header.stamp = *qdata.rcl_stamp;

      map_pub_->publish(*pc2_msg);
    }
    {
      // publish map and number of observations of each point
      auto pc2_msg = std::make_shared<PointCloudMsg>();
      pcl::PointCloud<pcl::PointXYZI> cloud;
      cloud.reserve(map->cloud.pts.size());

      auto pcitr = map->cloud.pts.begin();
      auto ititr = map->movabilities.begin();
      for (; pcitr != map->cloud.pts.end(); pcitr++, ititr++) {
        // remove recent points
        if (ititr->second < 10) continue;
        // remove points that are for sure dynamic
        if (((float)ititr->first / (float)ititr->second) > 0.5) continue;
        pcl::PointXYZI pt;
        pt.x = pcitr->x;
        pt.y = pcitr->y;
        pt.z = pcitr->z;
        pt.intensity = ((float)ititr->first / (float)ititr->second);
        cloud.points.push_back(pt);
      }

      pcl::toROSMsg(cloud, *pc2_msg);
      pc2_msg->header.frame_id = "localization keyframe";
      pc2_msg->header.stamp = *qdata.rcl_stamp;

      movability_map_pub_->publish(*pc2_msg);
    }
  }

  // output
  qdata.current_map_loc = map;
}

void WindowedMapRecallModule::visualizeImpl(QueryCache &qdata, MapCache &,
                                            const Graph::ConstPtr &,
                                            std::mutex &) {}

}  // namespace lidar
}  // namespace tactic
}  // namespace vtr