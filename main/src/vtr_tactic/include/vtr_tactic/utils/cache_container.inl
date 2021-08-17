/**
 * \file cache_container.inl
 * \brief Implements the reference-needed template functions.
 * \details They are hidden in this file to allow the cache classes to be
 * declared with incomplete types.
 *
 * \author Autonomous Space Robotics Lab (ASRL)
 */
#pragma once

#include <vtr_tactic/utils/cache_container.hpp>

namespace vtr {
namespace common {

template <typename Type>
std::ostream &cache_accessor<Type>::operator<<(std::ostream &os) const {
  // return os << *this;
  name_.empty() ? os << "(noname)" : os << name_;
  os << ": ";
  using namespace safe_stream::op;
  is_valid() ? os << operator*()
             : os << "(empty!)";  // safe_stream << in case T doesn't have <<
  return os;
}

// template <typename Type, bool Guaranteed>
// auto cache_ptr<Type, Guaranteed>::operator=(Type &&datum) -> my_t & {
//   return assign(std::move(datum));
// }
// template <typename Type, bool Guaranteed>
// auto cache_ptr<Type, Guaranteed>::operator=(const Type &datum) -> my_t & {
//   return assign(datum);
// }

// template <typename Type, bool Guaranteed>
// auto cache_ptr<Type, Guaranteed>::fallback(const Type &datum) -> my_t & {
//   return fallback<const Type &>(datum);
// }
// template <typename Type, bool Guaranteed>
// auto cache_ptr<Type, Guaranteed>::fallback(Type &&datum) -> my_t & {
//   return fallback<Type &&>(std::move(datum));
// }

}  // namespace common
}  // namespace vtr
