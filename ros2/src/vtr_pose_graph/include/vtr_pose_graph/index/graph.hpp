#pragma once

#include <mutex>
#include <vtr_pose_graph/index/callback_interface.hpp>
#include <vtr_pose_graph/index/graph_base.hpp>

namespace vtr {
namespace pose_graph {

template <class V, class E, class R>
class Graph : public virtual GraphBase<V, E, R> {
 public:
  using Base = GraphBase<V, E, R>;
  using RType = GraphBase<V, E, R>;

  using Base::edges_;
  using Base::graph_;
  using Base::id_;
  using Base::run;
  using Base::runs_;
  using Base::vertices_;
  using IdType = typename Base::IdType;

  // We have to manually import the typedefs, as they exist in dependent scope
  // and the compiler cannot find them by default
  using RunType = typename Base::RunType;
  using VertexType = typename Base::VertexType;
  using EdgeType = typename Base::EdgeType;

  using VertexPtr = typename Base::VertexPtr;
  using EdgePtr = typename Base::EdgePtr;
  using RunPtr = typename Base::RunPtr;

  using RunIdType = typename Base::RunIdType;
  using VertexIdType = typename Base::VertexIdType;
  using EdgeIdType = typename Base::EdgeIdType;
  using EdgeTypeEnum = typename Base::EdgeTypeEnum;
  using SimpleVertexId = typename Base::SimpleVertexId;
  using SimpleEdgeId = typename Base::SimpleEdgeId;

  using RunMap = typename Base::RunMap;
  using VertexMap = typename Base::VertexMap;
  using EdgeMap = typename Base::EdgeMap;
#if 0
  using TransformType = typename EdgeType::TransformType;
#endif
  using CallbackPtr = typename CallbackInterface<V, E, R>::Ptr;

  using UniqueLock = std::unique_lock<std::recursive_mutex>;
  using LockGuard = std::lock_guard<std::recursive_mutex>;

  PTR_TYPEDEFS(Graph)

  /**
   * \brief Pseudo-constructor to make shared pointers
   */
  static Ptr MakeShared();
  static Ptr MakeShared(const IdType& id);

  Graph();
  /**
   * \brief Construct an empty graph with an id
   */
  Graph(const IdType& id);
  Graph(const Graph&) = default;
  /**
   * \brief Move constructor (manually implemented due to virtual inheritance)
   */
  Graph(Graph&& other);

  Graph& operator=(const Graph&) = default;
  /**
   * \brief Move assignment (manually implemented due to virtual inheritance)
   */
  Graph& operator=(Graph&& other);
#if 0
  /**
   * \brief Set the callback handling procedure
   */
  void setCallbackMode(
      const CallbackPtr& manager = CallbackPtr(new IgnoreCallbacks<V, E, R>()));

  /**
   * \brief Get a pointer to the callback manager
   */
  inline const CallbackPtr& callbacks() const { return callbackManager_; }
#endif
  /**
   * \brief Add a new run an increment the run id
   */
  virtual RunIdType addRun();

  /**
   * \brief Return a blank vertex (current run) with the next available Id
   */
  virtual VertexPtr addVertex();

  /**
   * \brief Return a blank vertex with the next available Id
   */
  virtual VertexPtr addVertex(const RunIdType& runId);

  /**
   * \brief Return a blank edge with the next available Id
   */
  virtual EdgePtr addEdge(const VertexIdType& from, const VertexIdType& to,
                          const EdgeTypeEnum& type = EdgeTypeEnum::Temporal,
                          bool manual = false);
#if 0
  /**
   * \brief Return a blank edge with the next available Id
   */
  virtual EdgePtr addEdge(const VertexIdType& from, const VertexIdType& to,
                          const TransformType& T_to_from,
                          const EdgeTypeEnum& type = EdgeTypeEnum::Temporal,
                          bool manual = false);
#endif
  /**
   * \brief Callback functions that can be subclassed
   */
  virtual void runCallback() {}
  virtual void vertexCallback() {}
  virtual void edgeCallback() {}

  /**
   * \brief Acquire a lock object that blocks modifications
   */
  inline UniqueLock guard() { return UniqueLock(mtx_); }

  /**
   * \brief Manually lock the graph, preventing modifications
   */
  inline void lock() { mtx_.lock(); }

  /**
   * \brief Manually unlock the graph, allowing modifications
   */
  inline void unlock() { mtx_.unlock(); }

  /**
   * \brief Get a reference to the mutex
   */
  inline std::recursive_mutex& mutex() { return mtx_; }

 protected:
  /**
   * \brief The current run
   */
  RunPtr currentRun_;

  /**
   * \brief The current maximum run index
   */
  RunIdType lastRunIdx_;

  /**
   * \brief The current maximum run index
   */
  CallbackPtr callbackManager_;

  /**
   * \brief Used to lock changes to the graph during long-running operations
   */
  std::recursive_mutex mtx_;
};

using BasicGraph = Graph<VertexBase, EdgeBase, RunBase<VertexBase, EdgeBase>>;

}  // namespace pose_graph
}  // namespace vtr

#include <vtr_pose_graph/index/graph.inl>

#if 0
#if !defined(BASIC_GRAPH_NO_EXTERN) && defined(NDEBUG)
namespace asrl {
namespace pose_graph {

extern template class Graph<VertexBase, EdgeBase,
                            RunBase<VertexBase, EdgeBase>>;

EVAL_TYPED_DECLARE_EXTERN(double, BasicGraph)
EVAL_TYPED_DECLARE_EXTERN(bool, BasicGraph)

}  // namespace pose_graph
}  // namespace asrl
#endif
#endif