//////////////////////////////////////////////////////////////////////////////////////////////
/// \file LandmarkStateVar.hpp
///
/// \author Sean Anderson, ASRL
//////////////////////////////////////////////////////////////////////////////////////////////

#ifndef STEAM_LANDMARK_STATE_VARIABLE_HPP
#define STEAM_LANDMARK_STATE_VARIABLE_HPP

#include <steam/state/StateVariable.hpp>
#include <steam/evaluator/blockauto/transform/TransformEvaluator.hpp>

namespace steam {
namespace se3 {

/////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Landmark state variable
/////////////////////////////////////////////////////////////////////////////////////////////
class LandmarkStateVar : public StateVariable<Eigen::Vector4d>
{
 public:

  /// Convenience typedefs
  typedef boost::shared_ptr<LandmarkStateVar> Ptr;
  typedef boost::shared_ptr<const LandmarkStateVar> ConstPtr;

  //////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Constructor from a global 3D point
  //////////////////////////////////////////////////////////////////////////////////////////////
  LandmarkStateVar(const Eigen::Vector3d& v_0);

  //////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Destructor
  //////////////////////////////////////////////////////////////////////////////////////////////
  virtual ~LandmarkStateVar() {}

  //////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Update the landmark state from the 3-dimensional perturbation
  //////////////////////////////////////////////////////////////////////////////////////////////
  virtual bool update(const Eigen::VectorXd& perturbation);

  //////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Clone method
  //////////////////////////////////////////////////////////////////////////////////////////////
  virtual StateVariableBase::Ptr clone() const;

  //////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Set value -- mostly for landmark initialization
  //////////////////////////////////////////////////////////////////////////////////////////////
  void set(const Eigen::Vector3d& v);

 private:

  //////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Refresh the homogeneous scaling
  //////////////////////////////////////////////////////////////////////////////////////////////
  void refreshHomogeneousScaling();

};

} // se3
} // steam

#endif // STEAM_LANDMARK_STATE_VARIABLE_HPP