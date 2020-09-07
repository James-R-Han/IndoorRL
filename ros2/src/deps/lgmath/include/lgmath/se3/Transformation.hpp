//////////////////////////////////////////////////////////////////////////////////////////////
/// \file Transformation.hpp
/// \brief Header file for a transformation matrix class.
/// \details Light weight transformation class, intended to be fast, and not to provide
///          unnecessary functionality.
///
/// \author Sean Anderson
//////////////////////////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////////////////////////
///
/// A note on EIGEN_MAKE_ALIGNED_OPERATOR_NEW (Sean Anderson, as of May 23, 2013)
/// (also see http://eigen.tuxfamily.org/dox-devel/group__TopicStructHavingEigenMembers.html)
///
/// Fortunately, Eigen::Matrix3d and Eigen::Vector3d are NOT 16-byte vectorizable,
/// therefore this class should not require alignment, and can be used normally in STL.
///
/// To inform others of the issue, classes that include *fixed-size vectorizable Eigen types*,
/// see http://eigen.tuxfamily.org/dox-devel/group__TopicFixedSizeVectorizable.html,
/// must include the above macro! Furthermore, special considerations must be taken if
/// you want to use them in STL containers, such as std::vector or std::map.
/// The macro overloads the dynamic "new" operator so that it generates
/// 16-byte-aligned pointers, this MUST be in the public section of the header!
///
//////////////////////////////////////////////////////////////////////////////////////////////

#ifndef LGM_TRANSFORMATION_HPP
#define LGM_TRANSFORMATION_HPP

#include <Eigen/Dense>

namespace lgmath {
namespace se3 {

class Transformation
{
 public:

  //////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Default constructor
  //////////////////////////////////////////////////////////////////////////////////////////////
  Transformation();

  //////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Copy constructor. Default implementation.
  //////////////////////////////////////////////////////////////////////////////////////////////
  Transformation(const Transformation&) = default;

  //////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Move constructor.  Manually implemented as Eigen doesn't support moving.
  //////////////////////////////////////////////////////////////////////////////////////////////
  Transformation(Transformation&& T);

  //////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Constructor
  //////////////////////////////////////////////////////////////////////////////////////////////
  explicit Transformation(const Eigen::Matrix4d& T);

  //////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Constructor. The transformation will be T_ba = [C_ba, -C_ba*r_ba_ina; 0 0 0 1]
  //////////////////////////////////////////////////////////////////////////////////////////////
  Transformation(const Eigen::Matrix3d& C_ba, const Eigen::Vector3d& r_ba_ina);

  //////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Constructor. The transformation will be T_ba = vec2tran(xi_ab)
  //////////////////////////////////////////////////////////////////////////////////////////////
  Transformation(const Eigen::Matrix<double,6,1>& xi_ab, unsigned int numTerms = 0);

  //////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Constructor. The transformation will be T_ba = vec2tran(xi_ab), xi_ab must be 6x1
  //////////////////////////////////////////////////////////////////////////////////////////////
  // explicit because you want operator*(Eigen::Vector4d) and operator*(this) --- ambiguous
  explicit Transformation(const Eigen::VectorXd& xi_ab);

  //////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Destructor. Default implementation.
  //////////////////////////////////////////////////////////////////////////////////////////////
  ~Transformation() = default;

  //////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Copy assignment operator. Default implementation.
  //////////////////////////////////////////////////////////////////////////////////////////////
  virtual Transformation& operator=(const Transformation&);

  //////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Move assignment operator. Manually implemented as Eigen doesn't support moving.
  //////////////////////////////////////////////////////////////////////////////////////////////
  virtual Transformation& operator=(Transformation&& T);

  //////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Gets basic matrix representation of the transformation
  //////////////////////////////////////////////////////////////////////////////////////////////
  Eigen::Matrix4d matrix() const;

  //////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Gets the underlying rotation matrix
  //////////////////////////////////////////////////////////////////////////////////////////////
  const Eigen::Matrix3d& C_ba() const;

  //////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Gets the "forward" translation r_ba_ina = -C_ba.transpose()*r_ab_inb
  //////////////////////////////////////////////////////////////////////////////////////////////
  Eigen::Vector3d r_ba_ina() const;

  //////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Gets the underlying r_ab_inb vector.
  //////////////////////////////////////////////////////////////////////////////////////////////
  const Eigen::Vector3d& r_ab_inb() const;

  //////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Get the corresponding Lie algebra using the logarithmic map
  //////////////////////////////////////////////////////////////////////////////////////////////
  Eigen::Matrix<double,6,1> vec() const;

  //////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Get the inverse matrix
  //////////////////////////////////////////////////////////////////////////////////////////////
  Transformation inverse() const;

  //////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Get the 6x6 adjoint transformation matrix
  //////////////////////////////////////////////////////////////////////////////////////////////
  Eigen::Matrix<double,6,6> adjoint() const;

  //////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Reproject the transformation matrix back onto SE(3). Setting force to false
  ///        triggers a conditional reproject that only happens if the determinant is of the
  ///        rotation matrix is poor; this is more efficient than always performing it.
  //////////////////////////////////////////////////////////////////////////////////////////////
  void reproject(bool force = true);

  //////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief In-place right-hand side multiply T_rhs
  //////////////////////////////////////////////////////////////////////////////////////////////
  virtual Transformation& operator*=(const Transformation& T_rhs);

  //////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Right-hand side multiply T_rhs
  //////////////////////////////////////////////////////////////////////////////////////////////
  virtual Transformation operator*(const Transformation& T_rhs) const;

  //////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief In-place right-hand side multiply this matrix by the inverse of T_rhs
  //////////////////////////////////////////////////////////////////////////////////////////////
  virtual Transformation& operator/=(const Transformation& T_rhs);

  //////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Right-hand side multiply this matrix by the inverse of T_rhs
  //////////////////////////////////////////////////////////////////////////////////////////////
  virtual Transformation operator/(const Transformation& T_rhs) const;

  //////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Right-hand side multiply this matrix by the homogeneous vector p_a
  //////////////////////////////////////////////////////////////////////////////////////////////
  Eigen::Vector4d operator*(const Eigen::Ref<const Eigen::Vector4d>& p_a) const;

 private:

  //////////////////////////////////////////////////////////////////////////////////////////////
  /// Rotation matrix from a to b
  //////////////////////////////////////////////////////////////////////////////////////////////
  Eigen::Matrix3d C_ba_;

  //////////////////////////////////////////////////////////////////////////////////////////////
  /// Translation vector from b to a, expressed in frame b
  //////////////////////////////////////////////////////////////////////////////////////////////
  Eigen::Vector3d r_ab_inb_;
};

} // se3
} // lgmath

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief print transformation
//////////////////////////////////////////////////////////////////////////////////////////////
std::ostream& operator<<(std::ostream& out, const lgmath::se3::Transformation& T);

#endif // LGM_TRANSFORMATION_HPP