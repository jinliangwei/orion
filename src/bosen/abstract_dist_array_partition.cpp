#include <glog/logging.h>
#include <algorithm>

#include <orion/bosen/abstract_dist_array_partition.hpp>
#include <orion/bosen/dist_array.hpp>
#include <orion/bosen/key.hpp>
#include <orion/bosen/julia_evaluator.hpp>

namespace orion {
namespace bosen {

AbstractDistArrayPartition::AbstractDistArrayPartition(
    DistArray* dist_array,
    const Config &config,
    type::PrimitiveType value_type,
    JuliaThreadRequester *julia_requester):
    kConfig(config),
    kValueType(value_type),
    dist_array_(dist_array),
    julia_requester_(julia_requester) { }

const std::string &
AbstractDistArrayPartition::GetDistArraySymbol() {
  return dist_array_->GetMeta().GetSymbol();
}

std::vector<int64_t>&
AbstractDistArrayPartition::GetDims() { return dist_array_->GetDims(); }

void
AbstractDistArrayPartition::ParseText(Blob *max_key,
                                      size_t line_number_start) {
  LOG(INFO) << __func__ << " on partition " << (void*) this;
  jl_value_t *value = nullptr, *dist_array_value_type = nullptr;

  JL_GC_PUSH2(&value, &dist_array_value_type);
  key_buff_.clear();
  auto &dist_array_meta = dist_array_->GetMeta();
  bool flatten_results = dist_array_meta.IsFlattenResults();
  const std::string &dist_array_sym = dist_array_meta.GetSymbol();
  JuliaModule map_func_module = dist_array_meta.GetMapFuncModule();
  const std::string &map_func_name = dist_array_meta.GetMapFuncName();
  size_t num_dims = dist_array_meta.GetNumDims();

  JuliaEvaluator::GetDistArrayValueType(dist_array_sym,
                                        reinterpret_cast<jl_datatype_t**>(&dist_array_value_type));

  auto map_type = dist_array_->GetMeta().GetMapType();
  switch (map_type) {
    case DistArrayMapType::kMap:
      {
        size_t line_number = line_number_start;
        std::vector<int64_t> key(num_dims);
        char *line = strtok(char_buff_.data(), "\n");
        while (line != nullptr) {
          if (flatten_results) {
            key.clear();
            JuliaEvaluator::ParseStringWithLineNumberFlatten(line_number,
                                                          line,
                                                          map_func_module,
                                                          map_func_name.c_str(),
                                                          &key,
                                                          num_dims,
                                                          dist_array_value_type,
                                                          &value);

            AppendJuliaValueArray(value);
          } else {
            JuliaEvaluator::ParseStringWithLineNumber(line_number,
                                                      line,
                                                      map_func_module,
                                                      map_func_name.c_str(),
                                                      &key,
                                                      num_dims,
                                                      &value);
            AppendJuliaValue(value);
          }
          size_t i = 0;
          for (auto key_ith : key) {
            key_buff_.push_back(key_ith);
            *(reinterpret_cast<int64_t*>(max_key->data()) + i) = std::max(
                key_ith, *(reinterpret_cast<int64_t*>(max_key->data()) + i));
            i = (i + 1) % num_dims;
          }
          line = strtok(nullptr, "\n");
          line_number++;
        }
      }
      break;
    case DistArrayMapType::kMapFixedKeys:
      {
        size_t line_number = line_number_start;
        char *line = strtok(char_buff_.data(), "\n");
        CHECK(!flatten_results);
        while (line != nullptr) {
          JuliaEvaluator::ParseStringValueOnlyWithLineNumber(line_number,
                                                             line,
                                                             map_func_module,
                                                             map_func_name.c_str(),
                                                             num_dims,
                                                             &value);
          key_buff_.push_back(line_number);
          AppendJuliaValue(value);
          line = strtok(nullptr, "\n");
          line_number++;
        }
      }
      break;
    case DistArrayMapType::kMapValues:
      {
        char *line = strtok(char_buff_.data(), "\n");
        while (line != nullptr) {
          JuliaEvaluator::ParseStringValueOnly(line,
                                               map_func_module,
                                               map_func_name.c_str(),
                                               num_dims,
                                               &value);
          if (flatten_results)
            AppendJuliaValueArray(value);
          else
            AppendJuliaValue(value);
          line = strtok(nullptr, "\n");
        }
      }
      break;
    case DistArrayMapType::kMapValuesNewKeys:
      {
        std::vector<int64_t> key(num_dims);
        char *line = strtok(char_buff_.data(), "\n");
        while (line != nullptr) {
          if (flatten_results) {
            LOG(FATAL);
            key.clear();
            JuliaEvaluator::ParseStringFlatten(line,
                                               map_func_module,
                                               map_func_name.c_str(),
                                               &key,
                                               num_dims,
                                               dist_array_value_type,
                                               &value);
            AppendJuliaValueArray(value);
          } else {
            JuliaEvaluator::ParseString(line,
                                        map_func_module,
                                        map_func_name.c_str(),
                                        &key,
                                        num_dims,
                                        &value);
            AppendJuliaValue(value);
          }
          size_t i = 0;
          for (auto key_ith : key) {
            key_buff_.push_back(key_ith);
            *(reinterpret_cast<int64_t*>(max_key->data()) + i) = std::max(
                key_ith, *(reinterpret_cast<int64_t*>(max_key->data()) + i));
            i = (i + 1) % num_dims;
          }
          line = strtok(nullptr, "\n");
        }
      }
      break;
    case DistArrayMapType::kNoMap:
      {
        jl_value_t *string_jl = nullptr;
        JL_GC_PUSH1(&string_jl);
        char *line = strtok(char_buff_.data(), "\n");
        while (line != nullptr) {
          string_jl = jl_cstr_to_string(line);
          line = strtok(nullptr, "\n");
          AppendJuliaValue(value);
        }
        JL_GC_POP();
      }
      break;
    default:
      LOG(FATAL) << "shouldn't happend";
  }
  char_buff_.clear();
  JL_GC_POP();
}

bool
AbstractDistArrayPartition::LoadTextFile(const std::string &path,
                                         int32_t partition_id) {
  size_t offset = path.find_first_of(':');
  std::string prefix = path.substr(0, offset);
  std::string file_path = path.substr(offset + 3, path.length() - offset - 3);
  bool read = false;
  if (prefix == "hdfs") {
    read = LoadFromHDFS(kConfig.kHdfsNameNode, file_path, partition_id,
                        kConfig.kNumExecutors,
                        kConfig.kPartitionSizeMB * 1024 * 1024,
                        &char_buff_);
  } else if (prefix == "file") {
    read = LoadFromPosixFS(file_path, partition_id,
                           kConfig.kNumExecutors,
                           kConfig.kPartitionSizeMB * 1024 * 1024,
                           &char_buff_);
  } else {
    LOG(FATAL) << "Cannot parse the path specification " << path;
  }
  return read;
}

size_t
AbstractDistArrayPartition::CountNumLines() const {
  size_t num_lines = 0;
  for (auto c : char_buff_) {
    if (c == '\n') num_lines++;
  }
  return num_lines;
}

void
AbstractDistArrayPartition::AppendKeyValue(int64_t key, jl_value_t* value) {
  keys_.push_back(key);
  AppendJuliaValue(value);
}

void
AbstractDistArrayPartition::GetBufferBeginAndEnd(
    int32_t partition_id,
    size_t partition_size,
    size_t read_size,
    std::vector<char> *char_buff,
    size_t *begin,
    size_t *end) {
   // char_buff[(*end - 1)] is the last char to parse

  if (partition_id != 0) {
    for (*begin = 0; *begin < partition_size && *begin < read_size; (*begin)++) {
      if ((*char_buff)[*begin] == '\n') break;
    }
    *begin += 1;
  }
  CHECK_LT(*begin, partition_size);

  for (*end = std::min(partition_size, read_size); *end < read_size; (*end)++) {
    if ((*char_buff)[*end] == '\n') {
      *end += 1;
      break;
    }
  }
}

bool
AbstractDistArrayPartition::LoadFromHDFS(
    const std::string &hdfs_name_node,
    const std::string &file_path,
    int32_t partition_id,
    size_t num_executors,
    size_t partition_size,
    std::vector<char> *char_buff) {
#ifdef ORION_USE_HDFS
  size_t begin = 0, end = 0;
  hdfsFS fs = hdfsConnect(hdfs_name_node.c_str(), 0);
  CHECK(fs != NULL) << hdfs_name_node;
  hdfsFileInfo* data_info = hdfsGetPathInfo(fs, file_path.c_str());
  CHECK(data_info != NULL);
  size_t file_size = data_info->mSize;
  hdfsFreeFileInfo(data_info, 1);
  size_t read_offset = partition_id * partition_size;

  if (read_offset >= file_size) return false;
  size_t read_size = std::min(read_offset + partition_size * 2,
                              file_size - read_offset);

  std::vector<char> temp_char_buff(read_size + 1);
  hdfsFile data_file = hdfsOpenFile(fs, file_path.c_str(),
                                    O_RDONLY, 0, 0, 0);
  CHECK(data_file != NULL);

  size_t total_read_size = 0;
  const size_t kMaxReadSize = 64 * 1024 * 1024;
  while (total_read_size < read_size) {
    size_t curr_read_size = read_size - total_read_size;
    curr_read_size = (curr_read_size > kMaxReadSize) ? kMaxReadSize : curr_read_size;
    size_t read_count = hdfsPread(fs, data_file,
                                 read_offset + total_read_size,
                                 temp_char_buff.data() + total_read_size,
                                 curr_read_size);
    CHECK_GT(read_count, 0) << "read error! errno = " << errno;
    total_read_size += read_count;
  }
  CHECK_EQ(total_read_size, read_size)
      << " read_size = " << read_size
      << " file_size = " << file_size;
  hdfsCloseFile(fs, data_file);
  GetBufferBeginAndEnd(partition_id, partition_size,
                       read_size,
                       &temp_char_buff, &begin, &end);
  temp_char_buff[end] = '\0';

  char_buff->resize(end - begin + 1);
  memcpy(char_buff->data(), temp_char_buff.data() + begin, end - begin + 1);
  return true;
#else
  LOG(FATAL) << "HDFS is not supported in this build";
  return false;
#endif
}

bool
AbstractDistArrayPartition::LoadFromPosixFS(
    const std::string &file_path, int32_t partition_id,
    size_t num_executors, size_t partition_size,
    std::vector<char> *char_buff) {
  size_t begin = 0, end = 0;
  FILE *data_file = fopen(file_path.c_str(), "r");
  CHECK(data_file) << file_path << " open failed";
  fseek(data_file, 0, SEEK_END);
  size_t file_size = ftell(data_file);
  size_t read_offset = partition_id * partition_size;

  if (read_offset >= file_size) return false;
  size_t read_size = std::min(read_offset + partition_size * 2,
                              file_size - read_offset);
  fseek(data_file, read_offset, SEEK_SET);
  std::vector<char> temp_char_buff(read_size + 1);
  size_t read_count = fread(temp_char_buff.data(),
                            1, read_size,
                            data_file);

  CHECK_EQ(read_count, read_size)
      << "ferror = " << ferror(data_file)
      << " feof = " << feof(data_file)
      << " read_size = "
      << read_size
      << " read_count = "
      << read_count
      << " file_size = " << file_size;

  fclose(data_file);
  GetBufferBeginAndEnd(partition_id, partition_size,
                       read_size,
                       &temp_char_buff, &begin, &end);
  LOG(INFO) << "end = " << end;
  temp_char_buff[end] = '\0';

  char_buff->resize(end - begin + 1);
  memcpy(char_buff->data(), temp_char_buff.data() + begin, end - begin + 1);
  return true;
}

void
AbstractDistArrayPartition::Init(int64_t key_begin,
                                 size_t num_elements) {
  keys_.resize(num_elements);
  jl_value_t *init_values = nullptr,
              *dist_array_value_type = nullptr,
                  *values = nullptr;
  JL_GC_PUSH3(&init_values, &dist_array_value_type, &values);

  for (size_t i = 0; i < num_elements; i++) {
    keys_[i] = key_begin + i;
  }

  auto init_type = dist_array_->GetMeta().GetInitType();
  auto random_init_type = dist_array_->GetMeta().GetRandomInitType();
  switch (init_type) {
    case DistArrayInitType::kNormalRandom:
      {
        JuliaEvaluator::RandNormal(random_init_type,
                                   &init_values,
                                   num_elements);
      }
      break;
    case DistArrayInitType::kUniformRandom:
      {
        JuliaEvaluator::RandNormal(random_init_type,
                                &init_values,
                                num_elements);
      }
      break;
    default:
      LOG(FATAL) << "not yet supported " << static_cast<int>(init_type);
  }
  auto map_type = dist_array_->GetMeta().GetMapType();
  if (map_type != DistArrayMapType::kNoMap) {
    auto &dist_array_meta = dist_array_->GetMeta();
    const auto &dims = dist_array_meta.GetDims();
    auto map_func_module = dist_array_meta.GetMapFuncModule();
    const auto &map_func_name = dist_array_meta.GetMapFuncName();
    const auto &dist_array_sym = dist_array_meta.GetSymbol();
    JuliaEvaluator::GetDistArrayValueType(dist_array_sym,
                                       reinterpret_cast<jl_datatype_t**>(&dist_array_value_type));
    std::vector<int64_t> output_keys;
    JuliaEvaluator::RunMapGeneric(
        map_type,
        dims,
        dims,
        keys_.size(),
        keys_.data(),
        init_values,
        map_func_module,
        map_func_name,
        &output_keys,
        dist_array_value_type,
        &values);
    keys_ = output_keys;
    AppendJuliaValueArray(values);
  } else {
    AppendJuliaValueArray(init_values);
  }
  JL_GC_POP();
}

void
AbstractDistArrayPartition::Map(AbstractDistArrayPartition *child_partition) {
  jl_value_t *init_values = nullptr,
   *dist_array_value_type = nullptr,
            *input_values = nullptr,
           *output_values = nullptr;
  JL_GC_PUSH4(&init_values, &dist_array_value_type, &input_values, &output_values);

  GetJuliaValueArray(&input_values);
  auto *child_dist_array = child_partition->dist_array_;
  auto &dist_array_meta = child_dist_array->GetMeta();
  const auto &child_dims = dist_array_meta.GetDims();
  auto map_func_module = dist_array_meta.GetMapFuncModule();
  const auto &map_func_name = dist_array_meta.GetMapFuncName();
  const auto &dist_array_sym = dist_array_meta.GetSymbol();
  auto map_type = dist_array_meta.GetMapType();
  JuliaEvaluator::GetDistArrayValueType(dist_array_sym,
                                     reinterpret_cast<jl_datatype_t**>(&dist_array_value_type));
  std::vector<int64_t> output_keys;
  const auto &parent_dims = GetDims();
  JuliaEvaluator::RunMapGeneric(map_type,
                                parent_dims,
                                child_dims,
                                keys_.size(),
                                keys_.data(),
                                init_values,
                                map_func_module,
                                map_func_name,
                                &output_keys,
                                dist_array_value_type,
                                &output_values);
  child_partition->keys_ = output_keys;
  child_partition->AppendJuliaValueArray(output_values);
  JL_GC_POP();
}

void
AbstractDistArrayPartition::ComputeKeysFromBuffer(
    const std::vector<int64_t> &dims) {
  size_t num_dims = dims.size();
  keys_.clear();
  for (int i = 0; i < key_buff_.size(); i += num_dims) {
    int64_t key_i = key::array_to_int64(dims, key_buff_.data() + i);
    keys_.push_back(key_i);
  }
  key_buff_.clear();
}

void
AbstractDistArrayPartition::ComputeHashRepartitionIdsAndRepartition(size_t num_partitions) {
  std::vector<int32_t> repartition_ids(keys_.size());
  for (size_t i = 0; i < keys_.size(); i++) {
    int32_t repartition_id = keys_[i] % num_partitions;
    keys_[i] = repartition_id;
  }
  Repartition(repartition_ids.data());
}

void
AbstractDistArrayPartition::ComputeRepartitionIdsAndRepartition(
    const std::string &repartition_func_name) {
  jl_value_t *array_type = nullptr,
            *keys_vec_jl = nullptr,
            *dims_vec_jl = nullptr,
 *repartition_ids_vec_jl = nullptr;
  JL_GC_PUSH4(&array_type, &keys_vec_jl, &dims_vec_jl, &repartition_ids_vec_jl);
  auto &dims = dist_array_->GetDims();

  array_type = jl_apply_array_type(jl_int64_type, 1);
  keys_vec_jl = reinterpret_cast<jl_value_t*>(
      jl_ptr_to_array_1d(array_type, keys_.data(), keys_.size(), 0));
  dims_vec_jl = reinterpret_cast<jl_value_t*>(
      jl_ptr_to_array_1d(array_type, dims.data(), dims.size(), 0));

  jl_function_t *repartition_func = JuliaEvaluator::GetFunction(jl_main_module, repartition_func_name.c_str());
  repartition_ids_vec_jl = jl_call2(repartition_func, keys_vec_jl, dims_vec_jl);
  CHECK(!jl_exception_occurred()) << jl_typeof_str(jl_exception_occurred());
  int32_t *repartition_ids = reinterpret_cast<int32_t*>(jl_array_data(repartition_ids_vec_jl));
  Repartition(repartition_ids);
  JL_GC_POP();
}

void
AbstractDistArrayPartition::ComputePrefetchIndinces(
    const std::string &prefetch_batch_func_name,
    const std::vector<int32_t> &dist_array_ids_vec,
    PointQueryKeyDistArrayMap *point_key_vec_map,
    RangeQueryKeyDistArrayMap *range_key_vec_map) {
  jl_value_t **jl_values;
  JL_GC_PUSHARGS(jl_values, 7);

  jl_value_t* &dims_vec_jl = jl_values[0];
  jl_value_t* &keys_vec_jl = jl_values[1];
  jl_value_t* &values_vec_jl = jl_values[2];
  jl_value_t* &keys_array_type_jl = jl_values[3];
  jl_value_t* &ids_array_type_jl = jl_values[4];
  jl_value_t* &ids_array_jl = jl_values[5];
  jl_value_t* &ret_jl = jl_values[6];

  std::vector<int32_t> temp_dist_array_ids_vec = dist_array_ids_vec;

  auto &dims = GetDims();
  keys_array_type_jl = jl_apply_array_type(jl_int64_type, 1);
  ids_array_type_jl = jl_apply_array_type(jl_int32_type, 1);

  dims_vec_jl = reinterpret_cast<jl_value_t*>(jl_ptr_to_array_1d(
      keys_array_type_jl, dims.data(), dims.size(), 0));
  keys_vec_jl = reinterpret_cast<jl_value_t*>(jl_ptr_to_array_1d(
      keys_array_type_jl, keys_.data(), keys_.size(), 0));
  ids_array_jl = reinterpret_cast<jl_value_t*>(jl_ptr_to_array_1d(
      ids_array_type_jl, temp_dist_array_ids_vec.data(),
      temp_dist_array_ids_vec.size(), 0));

  GetJuliaValueArray(&values_vec_jl);
  jl_value_t *args[4];
  args[0] = keys_vec_jl;
  args[1] = values_vec_jl;
  args[2] = dims_vec_jl;
  args[3] = ids_array_jl;
  jl_function_t *prefetch_batch_func
      = JuliaEvaluator::GetFunction(jl_main_module,
                                    prefetch_batch_func_name.c_str());
  ret_jl = jl_call(prefetch_batch_func, args, 4);
  jl_value_t *point_key_dist_array_vec_jl =  jl_get_nth_field(ret_jl, 0);
  jl_value_t *range_key_dist_array_vec_jl = jl_get_nth_field(ret_jl, 1);

  for (size_t i = 0; i < dist_array_ids_vec.size(); i++) {
    int32_t dist_array_id = dist_array_ids_vec[0];
    {
      auto iter_pair = point_key_vec_map->emplace(dist_array_id, PointQueryKeyVec());
      auto iter = iter_pair.first;
      auto &key_vec = iter->second;
      jl_value_t *my_point_key_vec_jl = jl_arrayref(
          reinterpret_cast<jl_array_t*>(point_key_dist_array_vec_jl), i);
      size_t num_keys = jl_array_len(my_point_key_vec_jl);
      for (size_t j = 0; j < num_keys; j++) {
        jl_value_t *point_key_jl = jl_arrayref(
            reinterpret_cast<jl_array_t*>(my_point_key_vec_jl), j);
        int64_t point_key = jl_unbox_int64(point_key_jl);
        key_vec.push_back(point_key);
      }
    }

    {
      auto iter_pair = range_key_vec_map->emplace(dist_array_id, RangeQueryKeyVec());
      auto iter = iter_pair.first;
      auto &range_key_vec = iter->second;
      jl_value_t *my_range_key_vec_jl = jl_arrayref(
          reinterpret_cast<jl_array_t*>(range_key_dist_array_vec_jl), i);
      size_t num_keys = jl_array_len(my_range_key_vec_jl);
      for (size_t j = 0; j < num_keys; j++) {
        jl_value_t *range_key_tuple_jl = jl_arrayref(
            reinterpret_cast<jl_array_t*>(my_range_key_vec_jl), j);
        jl_value_t *range_key_jl = jl_get_nth_field(
            range_key_tuple_jl, 0);
        jl_value_t *range_num_keys_jl = jl_get_nth_field(
            range_key_tuple_jl, 1);
        int64_t point_key = jl_unbox_int64(range_key_jl);
        size_t num_keys = jl_unbox_uint64(range_num_keys_jl);
        range_key_vec.emplace_back(point_key, num_keys);
      }
    }
  }
  JL_GC_POP();
}

void
AbstractDistArrayPartition::Execute(
    const std::string &loop_batch_func_name) {
  LOG(INFO) << __func__ << " " << loop_batch_func_name << " start!";
  jl_value_t *dims_vec_jl = nullptr,
              *keys_vec_jl = nullptr,
           *values_vec_jl = nullptr,
      *dims_array_type_jl = nullptr,
      *keys_array_type_jl = nullptr;

  JL_GC_PUSH5(&dims_vec_jl,
              &keys_vec_jl,
              &values_vec_jl,
              &dims_array_type_jl,
              &keys_array_type_jl);

  auto &dims = GetDims();
  dims_array_type_jl = jl_apply_array_type(jl_int64_type, 1);
  keys_array_type_jl = jl_apply_array_type(jl_int64_type, 1);

  dims_vec_jl = reinterpret_cast<jl_value_t*>(jl_ptr_to_array_1d(
      dims_array_type_jl, dims.data(), dims.size(), 0));
  keys_vec_jl = reinterpret_cast<jl_value_t*>(jl_ptr_to_array_1d(
      keys_array_type_jl, keys_.data(), keys_.size(), 0));

  GetJuliaValueArray(&values_vec_jl);

  jl_function_t *exec_loop_func
      = JuliaEvaluator::GetFunction(jl_main_module,
                                    loop_batch_func_name.c_str());
  jl_call3(exec_loop_func, keys_vec_jl, values_vec_jl, dims_vec_jl);
  JuliaEvaluator::AbortIfException();
  JL_GC_POP();
  LOG(INFO) << __func__ << " " << loop_batch_func_name << " end!";
}

}
}
