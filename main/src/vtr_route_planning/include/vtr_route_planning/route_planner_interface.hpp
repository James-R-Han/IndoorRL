// Copyright 2021, Autonomous Space Robotics Lab (ASRL)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/**
 * \file route_planner_interface.hpp
 * \author Yuchen Wu, Autonomous Space Robotics Lab (ASRL)
 */
#pragma once

#include "vtr_tactic/types.hpp"

namespace vtr {
namespace route_planning {

class RoutePlannerInterface {
 public:
  PTR_TYPEDEFS(RoutePlannerInterface);
  using PathType = tactic::PathType;
  using VertexId = tactic::VertexId;

  virtual ~RoutePlannerInterface() = default;

  virtual PathType path(const VertexId& from, const VertexId& to) = 0;
  virtual PathType path(const VertexId& from, const VertexId::List& to,
                        std::list<uint64_t>& idx) = 0;
};

}  // namespace route_planning
}  // namespace vtr
