#pragma once

#include <stdint.h>

#include <orion/bosen/host_info.hpp>
#include <orion/constants.hpp>

namespace orion {

namespace bosen {

enum ConsistencyModel {
  // Stale synchronous parallel.
  SSP = 0,

  // SSP with server push
  // Assumes that all clients have the same number of bg threads.
  SSPPush = 1,

  SSPAggr = 2,

  LocalOOC = 6
};

enum UpdateSortPolicy {
  FIFO = 0,
  Random = 1,
  RelativeMagnitude = 2,
  FIFO_N_ReMag = 3,
  FixedOrder = 4
};

struct ServerTableInfo {
  int32_t staleness;
  size_t num_partitions;
  bool version_maintain;
  ConsistencyModel consistency_model;
  UpdateSortPolicy update_sort_policy;
  size_t recv_buff_capacity = 4*k1_Mi;
  size_t send_buff_capacity = 4*k1_Mi;
};

struct ExecutorTableInfo {
  int32_t staleness;
  size_t num_partitions_per_server;
  bool version_maintain;
  ConsistencyModel consistency_model;
  UpdateSortPolicy update_sort_policy;
  size_t recv_buff_capacity = 4*k1_Mi;
  size_t send_buff_capacity = 4*k1_Mi;
};

}  // namespace bosen

}
