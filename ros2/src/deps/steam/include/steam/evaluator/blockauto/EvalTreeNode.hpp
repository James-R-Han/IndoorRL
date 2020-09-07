//////////////////////////////////////////////////////////////////////////////////////////////
/// \file EvalTreeNode.hpp
///
/// \author Sean Anderson, ASRL
//////////////////////////////////////////////////////////////////////////////////////////////

#ifndef STEAM_EVAL_TREE_NODE_HPP
#define STEAM_EVAL_TREE_NODE_HPP

#include <steam/evaluator/blockauto/EvalTreeNodeBase.hpp>

#include <steam/evaluator/blockauto/OpenMpPool.hpp>

namespace steam {

//////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Class for a node in a block-automatic evaluation tree. While the true strength of
///        block automatic evaluation is in Jacobian evaluation, note that the most efficient
///        implementation involves first evaluating the nominal solution of the nonlinear
///        function at each level of the evaluator chain.
//////////////////////////////////////////////////////////////////////////////////////////////
template <typename TYPE>
class EvalTreeNode : public EvalTreeNodeBase
{
 public:

  //////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Default constructor
  //////////////////////////////////////////////////////////////////////////////////////////////
  EvalTreeNode();

  //////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Default constructor
  //////////////////////////////////////////////////////////////////////////////////////////////
  EvalTreeNode(const TYPE& value);

  //////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Destructor
  //////////////////////////////////////////////////////////////////////////////////////////////
  virtual ~EvalTreeNode() {}

  //////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Use when node was allocated using a pool. This function causes the object to
  ///        release itself back to the pool it was allocated from. While the user is
  ///        responsible for releasing the top level node, this interface method is required
  ///        in order for this node to release its children.
  //////////////////////////////////////////////////////////////////////////////////////////////
  virtual void release();

  //////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Release children and reset internals.
  //////////////////////////////////////////////////////////////////////////////////////////////
  virtual void reset();

  /////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Get current value
  /////////////////////////////////////////////////////////////////////////////////////////////
  const TYPE& getValue() const;

  /////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Set current value
  /////////////////////////////////////////////////////////////////////////////////////////////
  void setValue(const TYPE& value);

  /////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Static instance of pool type
  /////////////////////////////////////////////////////////////////////////////////////////////
  static OmpPool<EvalTreeNode<TYPE> > pool;

 private:

  /////////////////////////////////////////////////////////////////////////////////////////////
  /// \brief Instance of TYPE
  /////////////////////////////////////////////////////////////////////////////////////////////
  TYPE value_;

 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

} // steam

#include <steam/evaluator/blockauto/EvalTreeNode-inl.hpp>

#endif // STEAM_EVAL_TREE_NODE_HPP