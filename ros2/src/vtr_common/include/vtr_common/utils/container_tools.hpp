#pragma once

#include <deque>

namespace vtr {
namespace common {
namespace utils {

template <class C>
std::deque<std::reference_wrapper<typename C::value_type>> getRefs(
    C &container) {
  std::deque<std::reference_wrapper<typename C::value_type>> d;
  for (auto &&it : container) {
    d.emplace_back(it);
  }
  return d;
}

template <class C>
std::deque<std::reference_wrapper<const typename C::value_type>> getRefs(
    const C &container) {
  std::deque<std::reference_wrapper<const typename C::value_type>> d;
  for (auto &&it : container) {
    d.emplace_back(it);
  }
  return d;
}

}  // namespace utils
}  // namespace common
}  // namespace vtr