#pragma once

#include <orion/bosen/dist_array.hpp>
#include <orion/bosen/abstract_dist_array_partition.hpp>
#include <orion/bosen/dist_array_cache.hpp>
#include <orion/bosen/peer_recv_buffer.hpp>
#include <unordered_map>
#include <vector>
#include <orion/bosen/key_vec_type.hpp>
#include <glog/logging.h>
#include <julia.h>
#include <set>

namespace orion {
namespace bosen {
class AbstractExecForLoop {
 public:
  enum class RunnableStatus {
    kRunnable = 0,
      kPrefetchGlobalIndexedDistArrays = 1,
      kAwaitGlobalIndexedDistArrays = 2,
      kAwaitPredecessor = 3,
      kCompleted = 4,
      kSkip = 5
                };

  enum class PrefetchStatus {
    kNotPrefetched = 0,
      kPrefetchSent = 1,
      kPrefetchRecved = 2
                     };

 protected:
  PointQueryKeyDistArrayMap point_prefetch_dist_array_map_;
  size_t num_pending_prefetch_requests_ {0};
  PrefetchStatus prefetch_status_ {PrefetchStatus::kNotPrefetched};

  std::unordered_map<int32_t, DistArray*> space_partitioned_dist_arrays_;
  std::unordered_map<int32_t, DistArray*> time_partitioned_dist_arrays_;
  std::unordered_map<int32_t, DistArray*> global_indexed_dist_arrays_;
  std::unordered_map<int32_t, std::unique_ptr<AbstractDistArrayPartition>> dist_array_cache_;
  std::unordered_map<int32_t, DistArray*> dist_array_buffers_;
  std::set<int32_t> written_dist_array_ids_;

  DistArray *iteration_space_;
  const int32_t kExecutorId;
  const size_t kNumExecutors;
  const size_t kNumServers;
  const std::string kLoopBatchFuncName;
  const std::string kPrefetchBatchFuncName;
  bool curr_partition_prepared_ { false };
  bool dist_array_cache_prepared_ { false };
  uint8_t *time_partitions_serialized_bytes_ { nullptr };
  size_t time_partitions_serialized_size_ { 0 };
  AbstractDistArrayPartition* curr_partition_ { nullptr };

  ExecutorSendBufferMap buffer_send_buffer_map_;
  ExecutorSendBufferMap cache_send_buffer_map_;

  std::vector<std::string> accessed_dist_array_syms_;
  std::vector<std::string> global_read_only_var_vals_;
  std::vector<std::string> accumulator_var_syms_;

  std::vector<jl_value_t*> accessed_dist_arrays_;
  std::vector<jl_value_t*> global_read_only_var_jl_vals_;

 public:
  AbstractExecForLoop(
      int32_t executor_id,
      size_t num_executors,
      size_t num_servers,
      int32_t iteration_space_id,
      const int32_t *space_partitioned_dist_array_ids,
      size_t num_space_partitioned_dist_arrays,
      const int32_t *time_partitioned_dist_array_ids,
      size_t num_time_partitioned_dist_arrays,
      const int32_t *global_indexed_dist_array_ids,
      size_t num_gloabl_indexed_dist_arrays,
      const int32_t *dist_array_buffer_ids,
      size_t num_dist_array_buffers,
      const int32_t *written_dist_array_ids,
      size_t num_written_dist_array_ids,
      const int32_t *accessed_dist_array_ids,
      size_t num_accessed_dist_arrays,
      const std::string * const *global_read_only_var_vals,
      size_t num_global_read_only_var_vals,
      const std::string * const *accumulator_var_syms,
      size_t num_accumulator_var_syms,
      const char* loop_batch_func_name,
      const char *prefetch_batch_func_name,
      std::unordered_map<int32_t, DistArray> *dist_arrays,
      std::unordered_map<int32_t, DistArray> *dist_array_buffers);

  virtual ~AbstractExecForLoop();

  void Init();
  void Clear();

  virtual void FindNextToExecPartition() = 0;
  virtual RunnableStatus GetCurrPartitionRunnableStatus() = 0;
  // assume the current partition has been executed or skipped
  virtual int32_t GetTimePartitionIdToSend() = 0;
  virtual void ApplyPredecessorNotice(uint64_t clock) = 0;
  virtual int32_t GetSuccessorToNotify() = 0;
  virtual int32_t GetPredecessor() = 0;
  virtual uint64_t GetNoticeToSuccessor() = 0;
  virtual void PrepareToExecCurrPartition() = 0;
  virtual void ClearCurrPartition() = 0;

  bool SendGlobalIndexedDistArrays() const { return !global_indexed_dist_arrays_.empty(); }
  void SentAllPrefetchRequests();
  bool HasSentAllPrefetchRequests() const;
  bool HasRecvedAllPrefetches() const;
  bool HasRecvedAllTimePartitionedDistArrays(int32_t time_partition_id) const;

  void ComputePrefetchIndinces();
  void ExecuteForLoopPartition();
  void ClearSendBuffer();
  void SerializeAndClearPrefetchIds(ExecutorSendBufferMap *send_buffer_map);
  void CachePrefetchDistArrayValues(PeerRecvGlobalIndexedDistArrayDataBuffer **buff_vec,
                                    size_t num_buffs);
  void PrepareDistArrayCachePartitions();

  void SerializeAndClearGlobalPartitionedDistArrays();
  void SerializeAndClearDistArrayBuffers();
  void SerializeAndClearDistArrayCaches();
  void GetAndClearDistArrayBufferSendMap(ExecutorSendBufferMap *buffer_send_buffer_map);
  void GetAndClearDistArrayCacheSendMap(ExecutorSendBufferMap *cache_send_buffer_map);

  void SerializeAndClearPipelinedTimePartitions();
  void DeserializePipelinedTimePartitions(const uint8_t *bytes);
  void DeserializePipelinedTimePartitionsBuffVec(PeerRecvPipelinedTimePartitionsBuffer** buff_vec,
                                                 size_t num_buffs);
  SendDataBuffer GetAndResetSerializedTimePartitions();
};
}
}
