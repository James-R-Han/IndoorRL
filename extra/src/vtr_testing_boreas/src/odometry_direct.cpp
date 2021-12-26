#include <filesystem>

#include "rclcpp/rclcpp.hpp"

#include "vtr_common/timing/time_utils.hpp"
#include "vtr_common/utils/filesystem.hpp"
#include "vtr_logging/logging_init.hpp"
#include "vtr_navigation/navigator.hpp"

using namespace vtr;
using namespace vtr::common;
using namespace vtr::logging;
using namespace vtr::navigation;
using namespace vtr::tactic;

namespace {

float getFloatFromByteArray(char *byteArray, uint index) {
  return *((float *)(byteArray + index));
}

int64_t getStampFromPath(const std::string &path) {
  std::vector<std::string> parts;
  boost::split(parts, path, boost::is_any_of("/"));
  std::string stem = parts[parts.size() - 1];
  boost::split(parts, stem, boost::is_any_of("."));
  int64_t time1 = std::stoll(parts[0]);
  return time1;
}

// Input is a .bin binary file.
std::pair<int64_t, Eigen::MatrixXd> load_lidar(const std::string &path) {
  std::ifstream ifs(path, std::ios::binary);
  std::vector<char> buffer(std::istreambuf_iterator<char>(ifs), {});
  uint float_offset = 4;
  uint fields = 6;  // x, y, z, i, r, t
  uint point_step = float_offset * fields;
  uint N = floor(buffer.size() / point_step);
  Eigen::MatrixXd pc(Eigen::MatrixXd::Ones(N, fields));
  for (uint i = 0; i < N; ++i) {
    uint bufpos = i * point_step;
    for (uint j = 0; j < fields; ++j) {
      pc(i, j) =
          getFloatFromByteArray(buffer.data(), bufpos + j * float_offset);
    }
  }
  // Add offset to timestamps
  const auto time_micro = getStampFromPath(path);
  double t = double(time_micro) * 1.0e-6;
  pc.block(0, 5, N, 1).array() += t;

  return std::make_pair<int64_t, Eigen::MatrixXd>(time_micro * 1e3,
                                                  std::move(pc));
}

EdgeTransform load_T_robot_lidar(const std::string &path) {
  std::ifstream ifs(path, std::ios::in);

  Eigen::Matrix4d T_robot_lidar_mat;
  for (size_t row = 0; row < 4; row++)
    for (size_t col = 0; col < 4; col++) ifs >> T_robot_lidar_mat(row, col);

  Eigen::Matrix4d yfwd2xfwd;
  yfwd2xfwd << 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1;

  EdgeTransform T_robot_lidar(Eigen::Matrix4d(yfwd2xfwd * T_robot_lidar_mat),
                              Eigen::Matrix<double, 6, 6>::Zero());

  return T_robot_lidar;
}

}  // namespace

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  auto node = rclcpp::Node::make_shared("navigator");

  // Input directory sequence
  const auto odo_dir_str =
      node->declare_parameter<std::string>("odo_dir", "/tmp");
  fs::path odo_dir{utils::expand_user(utils::expand_env(odo_dir_str))};

  // Output directory
  const auto data_dir_str =
      node->declare_parameter<std::string>("data_dir", "/tmp");
  fs::path data_dir{utils::expand_user(utils::expand_env(data_dir_str))};

  // Configure logging
  const auto log_to_file = node->declare_parameter<bool>("log_to_file", false);
  const auto log_debug = node->declare_parameter<bool>("log_debug", false);
  const auto log_enabled = node->declare_parameter<std::vector<std::string>>(
      "log_enabled", std::vector<std::string>{});
  std::string log_filename;
  if (log_to_file) {
    // Log into a subfolder of the data directory (if requested to log)
    auto log_name = "vtr-" + timing::toIsoFilename(timing::clock::now());
    log_filename = data_dir / (log_name + ".log");
  }
  configureLogging(log_filename, log_debug, log_enabled);

  LOG(WARNING) << "Odometry Directory: " << odo_dir.string();
  LOG(WARNING) << "Output Directory: " << data_dir.string();

  // Pose graph
  auto graph = tactic::Graph::MakeShared((data_dir / "graph").string(), false);

  // Pipeline
  auto pipeline_factory = std::make_shared<ROSPipelineFactory>(node);
  pipeline_factory->add<lidar::LidarPipeline>();
  auto pipeline = pipeline_factory->make("pipeline");

  // Tactic
  auto tactic = std::make_shared<Tactic>(Tactic::Config::fromROS(node), node,
                                         pipeline, graph);

  tactic->setPipeline(PipelineMode::Branching);
  tactic->addRun();

  // Frame and transforms
  std::string robot_frame = "robot";
  std::string lidar_frame = "velodyne";

  const auto T_robot_lidar =
      load_T_robot_lidar(odo_dir / "calib" / "T_applanix_lidar.txt");
  const auto T_lidar_robot = T_robot_lidar.inverse();
  LOG(WARNING) << "Transform from " << robot_frame << " to " << lidar_frame
               << " has been set to" << T_lidar_robot;

  auto tf_static_broadcaster =
      std::make_shared<tf2_ros::StaticTransformBroadcaster>(node);
  auto msg = tf2::eigenToTransform(Eigen::Affine3d(T_robot_lidar.matrix()));
  msg.header.frame_id = robot_frame;
  msg.child_frame_id = lidar_frame;
  tf_static_broadcaster->sendTransform(msg);

  // List of point cloud data
  std::vector<std::filesystem::directory_entry> dirs;
  for (const auto &dir_entry :
       std::filesystem::directory_iterator{odo_dir / "lidar"}) {
    dirs.push_back(dir_entry);
  }
  std::sort(dirs.begin(), dirs.end());

  // Load dataset
  int i = 0;
  for (auto it = dirs.begin(); it != dirs.end(); it++, i++) {
    if (!rclcpp::ok()) break;

    // Load
    const auto [timestamp, points] = load_lidar(it->path().string());

    LOG(WARNING) << "Loading point cloud frame " << i << " with timestamp "
                 << timestamp << " with number of points " << points.rows();

    // Convert message to query_data format and store into query_data
    auto query_data = std::make_shared<lidar::LidarQueryCache>();

    // set timestamp
    query_data->stamp.emplace(timestamp);

    // put in the pointcloud msg pointer into query data
    query_data->points = std::move(points);

    // fill in the vehicle to sensor transform and frame names
    query_data->robot_frame.emplace(robot_frame);
    query_data->lidar_frame.emplace(lidar_frame);
    query_data->T_s_r.emplace(T_lidar_robot);

    // execute the pipeline
    tactic->runPipeline(query_data);
  }

  std::this_thread::sleep_for(5s);
  LOG(WARNING) << "Saving pose graph and reset.";
  tactic.reset();
  graph->save();
  graph.reset();
  LOG(WARNING) << "Saving pose graph and reset. - done!";

  rclcpp::shutdown();
}