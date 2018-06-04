#pragma once

#include <orion/bosen/dist_array.hpp>
#include <orion/bosen/abstract_dist_array_partition.hpp>
#include <orion/bosen/dist_array_cache.hpp>
#include <orion/bosen/dist_array_buffer_info.hpp>
#include <orion/bosen/peer_recv_buffer.hpp>
#include <unordered_map>
#include <vector>
#include <orion/bosen/key_vec_type.hpp>
#include <glog/logging.h>
#include <julia.h>
#include <set>
#include <list>

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
      kPrefetchRecved = 2,
      kSkipPrefetch = 3
                     };

 protected:
  PointQueryKeyDistArrayMap point_prefetch_dist_array_map_;
  size_t num_pending_prefetch_requests_ {0};
  PrefetchStatus prefetch_status_;

  std::unordered_map<int32_t, DistArray*> space_partitioned_dist_arrays_;
  std::unordered_map<int32_t, DistArray*> time_partitioned_dist_arrays_;
  std::unordered_map<int32_t, DistArray*> global_indexed_dist_arrays_;
  std::unordered_map<int32_t, DistArray*> accessed_global_indexed_dist_arrays_;
  std::unordered_map<int32_t, std::unique_ptr<AbstractDistArrayPartition>> dist_array_cache_;
  std::unordered_map<int32_t, DistArray*> dist_array_buffers_;
  std::unordered_map<int32_t, std::set<int32_t>> skipped_time_partitioned_dist_array_id_map_;
  std::set<int32_t> written_dist_array_ids_;
  std::set<int32_t> read_only_global_indexed_dist_array_ids_;
  std::unordered_map<int32_t, bool> dist_array_cache_buffer_to_create_accessor_;

  std::unordered_map<int32_t, DistArrayBufferDelayInfo> dist_array_buffer_delay_info_;
  size_t num_elements_per_exec_ {0};
  size_t num_elements_to_exec_ {0};
  size_t curr_partition_num_elements_executed_ {0};
  size_t curr_partition_length_ {0};
  bool skipped_ {false};
  using ServerPrefetchKeyMap = std::unordered_map<int32_t, PointQueryKeyDistArrayMap>;
  struct ServerPrefetchInfo {
    const ServerPrefetchKeyMap kKeyMap;
    const std::vector<int32_t> kRePrefetchDistArrayIds;
    ServerPrefetchInfo(const ServerPrefetchKeyMap &key_map,
                       std::vector<int32_t> &&re_prefetch_dist_array_ids):
        kKeyMap(key_map),
        kRePrefetchDistArrayIds(re_prefetch_dist_array_ids) { }
  };
  std::list<ServerPrefetchInfo> server_prefetch_info_list_;
  std::list<ServerPrefetchInfo>::iterator server_prefetch_info_list_iter_;
  ServerPrefetchKeyMap server_prefetch_key_map_;
  ServerPrefetchKeyMap const * server_prefetch_key_map_ptr_;

  DistArray *iteration_space_;
  const int32_t kExecutorId;
  const size_t kNumExecutors;
  const size_t kNumServers;
  const std::string kLoopBatchFuncName;
  const std::string kPrefetchBatchFuncName;
  const std::unordered_map<int32_t, DistArrayBufferInfo> &kDistArrayBufferInfoMap;
  const bool kIsRepeated;
  uint8_t *time_partitions_serialized_bytes_ { nullptr };
  size_t time_partitions_serialized_size_ { 0 };
  AbstractDistArrayPartition* curr_partition_ { nullptr };

  ExecutorSendBufferMap buffer_send_buffer_map_;
  ExecutorSendBufferMap cache_send_buffer_map_;

  std::vector<std::string> accessed_dist_array_syms_;
  std::vector<std::string> accessed_dist_array_buffer_syms_;
  std::vector<std::string> global_read_only_var_vals_;
  std::vector<std::string> accumulator_var_syms_;

  std::vector<jl_value_t*> accessed_dist_arrays_;
  std::vector<jl_value_t*> accessed_dist_array_buffers_;
  std::vector<jl_value_t*> global_read_only_var_jl_vals_;
  bool is_first_time_ { true };
  bool time_partitions_cleared_ { true };
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
      std::unordered_map<int32_t, DistArray> *dist_array_buffers,
      const std::unordered_map<int32_t, DistArrayBufferInfo> &dist_array_buffer_info_map,
      bool is_repeated);

  virtual ~AbstractExecForLoop();

  void ResetGlobalReadOnlyVarVals(const std::string * const *global_read_only_var_vals,
                                  size_t num_global_read_only_var_vals);
  void InitOnCreation();
  void Clear();
  void InitEachExecution(bool is_first_time);
  void InitExecInterval();
  bool IsRepeated() const { return kIsRepeated; }
  // assume the current partition has been executed or skipped
  virtual int32_t GetSuccessorToNotify() = 0;
  virtual int32_t GetPredecessor() = 0;
  virtual uint64_t GetNoticeToSuccessor() = 0;

  RunnableStatus GetRunnableStatus();
  bool SendGlobalIndexedDistArrays() const {
    return !global_indexed_dist_arrays_.empty();
  }
  void SentAllPrefetchRequests();
  void ToSkipPrefetch();
  bool SkipPrefetch() const;
  bool HasSentAllPrefetchRequests() const;
  bool HasRecvedAllPrefetches() const;
  bool HasRecvedAllTimePartitionedDistArrays(int32_t time_partition_id) const;

  void ComputePrefetchIndices();
  void ExecuteForLoopPartition();
  void Skip();
  void ClearSendBuffer();
  void SerializeAndClearPrefetchIds(ExecutorSendBufferMap *send_buffer_map);
  void CachePrefetchDistArrayValues(PeerRecvGlobalIndexedDistArrayDataBuffer **buff_vec,
                                    size_t num_buffs);

  void SerializeAndClearGlobalPartitionedDistArrays();
  void GetAndClearDistArrayBufferSendMap(ExecutorSendBufferMap *buffer_send_buffer_map);
  void GetAndClearDistArrayCacheSendMap(ExecutorSendBufferMap *cache_send_buffer_map);

  void SerializeAndClearPipelinedTimePartitions();
  void DeserializePipelinedTimePartitions(uint8_t *bytes);
  void DeserializePipelinedTimePartitionsBuffVec(PeerRecvPipelinedTimePartitionsBuffer** buff_vec,
                                                 size_t num_buffs);
  SendDataBuffer GetAndResetSerializedTimePartitions();
  virtual int32_t GetTimePartitionIdToSend() = 0;

 protected:
  void InitPartitionToExec();
 private:
  void UpdateDistArrayBufferDelayInfo(size_t num_elements_executed,
                                      bool end_of_partition,
                                      bool last_partition);
  void UpdateDistArrayBufferDelayInfoWhenSkipLastPartition();
  void PrepareSpaceDistArrayPartitions();
  void PrepareTimeDistArrayPartitions();
  void PrepareDistArrayCacheBufferPartitions();
  void ClearSpaceDistArrayPartitions();
  void ClearTimeDistArrayPartitions();
  void ClearDistArrayCacheBufferPartitions();
  void SerializeAndClearDistArrayBuffers();
  void SerializeAndClearDistArrayCaches();
  virtual void InitClocks() = 0;
  virtual bool IsCompleted() = 0;
  virtual void AdvanceClock() = 0;
  virtual bool LastPartition() = 0;
  virtual bool ToClearTimePartition() = 0;
  virtual void ComputePartitionIdsAndFindPartitionToExecute() = 0;
  virtual void ApplyPredecessorNotice(uint64_t clock) = 0;
  virtual int32_t GetCurrSpacePartitionId() = 0;
  virtual int32_t GetCurrTimePartitionId() = 0;
  virtual bool SkipTimePartition() = 0;
  virtual bool AwaitPredecessorForGlobalIndexedDistArrays() = 0;
};
}
}
