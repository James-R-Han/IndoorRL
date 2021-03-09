#pragma once

#include <vtr_navigation/modules/base_module.hpp>
#include <vtr_navigation/visualize.hpp>
#include <vtr_vision/features/extractor/base_feature_extractor.hpp>
#include <vtr_vision/features/extractor/extractor_configs.hpp>

namespace vtr {
namespace navigation {

/**
 * \brief A module that converts images from RGB to grayscale or other forms,
 * and extracts features using surf/orb in parallel.
 * \details
 * requires: qdata.[rig_images]
 * outputs: qdata.[rig_images, rig_features]
 *
 * This module stores the converted images (gray scaled, color constant) in
 * extra channels of each image of qdata.rig_images.
 * The features corresponding to each channel are stored in qdata.rig_features.
 * Only stereo matched features are stored.
 */
class ConversionExtractionModule : public BaseModule {
 public:
  /** \brief Static module identifier. \todo change this to static_name */
  static constexpr auto type_str_ = "conversion_extraction";

  /** \brief Collection of config parameters */
  struct Config {
    std::string feature_type = "ASRL_GPU_SURF";

    vision::ORBConfiguration opencv_orb_params;
#if GPUSURF_ENABLED
    asrl::GpuSurfConfiguration gpu_surf_params;
    asrl::GpuSurfStereoConfiguration gpu_surf_stereo_params;
#endif

    /** \brief The collection of user requested image conversions. */
    std::vector<std::string> conversions;

    /**
     * \brief The collection of color constant weights.
     * \details The size of this vector must be greater or equal to the number
     * of requested color constant conversions.
     */
    std::vector<double> color_constant_weights;

    /** \brief Histogram equalization flag for color constant transformation. */
    bool color_constant_histogram_equalization;

    /** \brief Flag for displaying the raw detected features */
    bool visualize_raw_features = false;
  };

  ConversionExtractionModule(std::string name = type_str_)
      : BaseModule{name} {};

  void setConfig(std::shared_ptr<Config> &config);

 protected:
  /**
   * \brief Uses multi-threading to perform image conversion (i.e. RGB to
   * grayscale and CC),and feature extraction in parallel for each rig, channel
   * and camera.
   */
  void run(QueryCache &qdata, MapCache &,
           const std::shared_ptr<const Graph> &) override;

  /**
   * \brief Visualizes raw features that were extracted on all images and their
   * conversions.
   */
  void visualize(QueryCache &qdata, MapCache &,
                 const std::shared_ptr<const Graph> &,
                 std::mutex &vis_mtx) override;

 private:
  /** \brief Algorithm Configuration */
  std::shared_ptr<Config> config_;

  /** \brief Feature Extractor */
  std::shared_ptr<vision::BaseFeatureExtractor> extractor_;
};

}  // namespace navigation
}  // namespace vtr
