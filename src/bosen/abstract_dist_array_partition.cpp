#include <glog/logging.h>
#include <algorithm>
#include <random>
#include <fstream>

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

const std::vector<int64_t>&
AbstractDistArrayPartition::GetDims() const { return dist_array_->GetDims(); }

size_t
AbstractDistArrayPartition::GetLength() const {
  CHECK(storage_type_ == DistArrayPartitionStorageType::kKeyValueBuffer);
  return keys_.size();
}

void
AbstractDistArrayPartition::ComputeKeyDiffs(const std::vector<int64_t> &target_keys,
                                            std::vector<int64_t> *diff_keys) const {
  CHECK(storage_type_ == DistArrayPartitionStorageType::kKeyValueBuffer ||
        storage_type_ == DistArrayPartitionStorageType::kAccessor);
  diff_keys->clear();
  const std::vector<int64_t> *keys_ptr = &keys_;
  std::vector<int64_t> temp_keys;
  if (!sorted_) {
    temp_keys = keys_;
    std::sort(temp_keys.begin(), temp_keys.end());
    keys_ptr = &temp_keys;
  }

  for (auto target_key : target_keys) {
    bool found = std::binary_search(keys_ptr->begin(), keys_ptr->end(),
                                    target_key);
    if (!found) diff_keys->emplace_back(target_key);
  }
}

void
AbstractDistArrayPartition::ParseText(Blob *max_key,
                                      size_t line_number_start) {
  CHECK(storage_type_ == DistArrayPartitionStorageType::kKeyValueBuffer);
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
  {
    std::vector<char> empty_buff;
    char_buff_.swap(empty_buff);
  }
  ShrinkValueVecToFit();
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
AbstractDistArrayPartition::SaveAsTextFile(const std::string &id_str,
                                           const std::string &to_string_func_name,
                                           const std::string &file_path) {
  CHECK(storage_type_ == DistArrayPartitionStorageType::kKeyValueBuffer);
  jl_value_t* key_array_type = nullptr;
  jl_value_t* dim_array_jl = nullptr;
  jl_value_t* key_array_jl = nullptr;
  jl_value_t* value_array_jl = nullptr;
  jl_value_t* record_str_vec_jl = nullptr;
  JL_GC_PUSH5(&key_array_type, &dim_array_jl, &key_array_jl,
              &value_array_jl, &record_str_vec_jl);

  GetJuliaValueArray(&value_array_jl);
  std::vector<int64_t> dims = GetDims();

  key_array_type = jl_apply_array_type(reinterpret_cast<jl_value_t*>(jl_int64_type), 1);
  dim_array_jl = reinterpret_cast<jl_value_t*>(
      jl_ptr_to_array_1d(key_array_type, dims.data(), dims.size(), 0));
  key_array_jl = reinterpret_cast<jl_value_t*>(
      jl_ptr_to_array_1d(key_array_type, keys_.data(), keys_.size(), 0));

  jl_function_t *to_string_func = JuliaEvaluator::GetFunction(
      jl_main_module, to_string_func_name.c_str());

  record_str_vec_jl = jl_call3(to_string_func, dim_array_jl, key_array_jl,
                               value_array_jl);
  JuliaEvaluator::AbortIfException();
  auto &meta = dist_array_->GetMeta();
  const auto &symbol = meta.GetSymbol();
  std::string file_name = file_path + "/" + symbol + std::string(".partition.") + id_str;
  std::ofstream out_fs(file_name);

  size_t num_record_strs = jl_array_len(reinterpret_cast<jl_array_t*>(record_str_vec_jl));
  for (size_t i = 0; i < num_record_strs; i++) {
    jl_value_t* record_str_jl = jl_arrayref(reinterpret_cast<jl_array_t*>(record_str_vec_jl), i);
    const char* record_str = jl_string_ptr(record_str_jl);
    out_fs << record_str << std::endl;
  }
  out_fs.close();
  JL_GC_POP();
}

void
AbstractDistArrayPartition::AppendKeyValue(int64_t key, jl_value_t* value) {
  keys_.push_back(key);
  AppendJuliaValue(value);
}

std::vector<int64_t>&
AbstractDistArrayPartition::GetKeys() {
  return keys_;
}

void
AbstractDistArrayPartition::Repartition(
    const int32_t *repartition_ids) {
  auto &dist_array_meta = dist_array_->GetMeta();
  auto partition_scheme = dist_array_meta.GetPartitionScheme();
  if (partition_scheme == DistArrayPartitionScheme::kSpaceTime) {
    RepartitionSpaceTime(repartition_ids);
  } else {
    Repartition1D(repartition_ids);
  }
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
  size_t read_size = std::min(partition_size * 2,
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
  size_t read_size = std::min(partition_size * 2,
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
  temp_char_buff[end] = '\0';

  char_buff->resize(end - begin + 1);
  memcpy(char_buff->data(), temp_char_buff.data() + begin, end - begin + 1);
  return true;
}

void
AbstractDistArrayPartition::Init(int64_t key_begin,
                                 size_t num_elements) {
  CHECK(storage_type_ == DistArrayPartitionStorageType::kKeyValueBuffer);
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
        JuliaEvaluator::RandUniform(random_init_type,
                                    &init_values,
                                    num_elements);
      }
      break;
    case DistArrayInitType::kFill:
      {
        const auto &serialized_init_value = dist_array_->GetMeta().GetInitValue();
        JuliaEvaluator::Fill(serialized_init_value,
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
  CHECK(storage_type_ == DistArrayPartitionStorageType::kKeyValueBuffer);
  jl_value_t *dist_array_value_type = nullptr,
                      *input_values = nullptr,
                     *output_values = nullptr;
  JL_GC_PUSH3(&dist_array_value_type, &input_values, &output_values);

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
                                input_values,
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
AbstractDistArrayPartition::GroupBy(AbstractDistArrayPartition* child_partition) {
  CHECK(storage_type_ == DistArrayPartitionStorageType::kKeyValueBuffer);
  jl_value_t **jl_values;
  JL_GC_PUSHARGS(jl_values, 7);
  jl_value_t *&input_values_jl = jl_values[0];
  jl_value_t *&output_value_type = jl_values[1];
  jl_value_t *&key_array_type = jl_values[2];
  jl_value_t *&key_array_jl = jl_values[3];
  jl_value_t *&parent_dim_array_jl = jl_values[4];
  jl_value_t *&child_dim_array_jl = jl_values[5];
  jl_value_t *&output_tuple_jl = jl_values[6];

  GetJuliaValueArray(&input_values_jl);
  auto *child_dist_array = child_partition->dist_array_;
  auto &dist_array_meta = child_dist_array->GetMeta();
  std::vector<int64_t> child_dims = dist_array_meta.GetDims();
  auto map_func_module = dist_array_meta.GetMapFuncModule();
  const auto &map_func_name = dist_array_meta.GetMapFuncName();
  const auto &dist_array_sym = dist_array_meta.GetSymbol();
  std::vector<int64_t> parent_dims = GetDims();
  JuliaEvaluator::GetDistArrayValueType(dist_array_sym,
                                        reinterpret_cast<jl_datatype_t**>(&output_value_type));

  key_array_type = jl_apply_array_type(reinterpret_cast<jl_value_t*>(jl_int64_type), 1);
  parent_dim_array_jl = reinterpret_cast<jl_value_t*>(
      jl_ptr_to_array_1d(key_array_type, parent_dims.data(), parent_dims.size(), 0));
  child_dim_array_jl = reinterpret_cast<jl_value_t*>(
      jl_ptr_to_array_1d(key_array_type, child_dims.data(), child_dims.size(), 0));
  key_array_jl = reinterpret_cast<jl_value_t*>(
      jl_ptr_to_array_1d(key_array_type, keys_.data(), keys_.size(), 0));

  jl_function_t *map_func_jl = JuliaEvaluator::GetFunction(
      GetJlModule(map_func_module), map_func_name.c_str());

  {
    jl_value_t *args[5];
    args[0] = parent_dim_array_jl;
    args[1] = child_dim_array_jl;
    args[2] = key_array_jl;
    args[3] = input_values_jl;
    args[4] = output_value_type;
    output_tuple_jl = jl_call(map_func_jl, args, 5);
    JuliaEvaluator::AbortIfException();
  }

  jl_value_t *output_key_array_jl = jl_get_nth_field(output_tuple_jl, 0);
  jl_value_t *output_values_jl = jl_get_nth_field(output_tuple_jl, 1);

  size_t num_output_keys = jl_array_len(output_key_array_jl);
  child_partition->keys_.resize(num_output_keys);

  uint8_t *output_key_array = reinterpret_cast<uint8_t*>(jl_array_data(output_key_array_jl));
  memcpy(child_partition->keys_.data(), output_key_array, num_output_keys * sizeof(int64_t));

  size_t num_output_values = jl_array_len(reinterpret_cast<jl_array_t*>(output_values_jl));
  CHECK_EQ(num_output_values, num_output_keys);
  child_partition->AppendJuliaValueArray(output_values_jl);
  JL_GC_POP();
}

void
AbstractDistArrayPartition::ComputeKeysFromBuffer(
    const std::vector<int64_t> &dims) {
  size_t num_dims = dims.size();
  keys_.resize(key_buff_.size() / num_dims);

  for (int i = 0; i < key_buff_.size(); i += num_dims) {
    int64_t key_i = key::array_to_int64(dims, key_buff_.data() + i);
    CHECK_GE(key_i, 0) << " " << key_buff_[i] << " " << key_buff_[i + 1];
    keys_[i / num_dims] = key_i;
  }
  {
    std::vector<int64_t> empty_buff;
    key_buff_.swap(empty_buff);
  }
}

void
AbstractDistArrayPartition::ComputeModuloRepartitionIdsAndRepartition(size_t num_partitions) {
  CHECK(storage_type_ == DistArrayPartitionStorageType::kKeyValueBuffer);
  std::vector<int32_t> repartition_ids(keys_.size());
  for (size_t i = 0; i < keys_.size(); i++) {
    int32_t repartition_id = keys_[i] % num_partitions;
    repartition_ids[i] = repartition_id;
  }
  Repartition(repartition_ids.data());
}

void
AbstractDistArrayPartition::ComputeRepartitionIdsAndRepartition(
    const std::string &repartition_func_name) {
  CHECK(storage_type_ == DistArrayPartitionStorageType::kKeyValueBuffer);
  jl_value_t *array_type = nullptr,
            *keys_vec_jl = nullptr,
            *dims_vec_jl = nullptr,
 *repartition_ids_vec_jl = nullptr;
  JL_GC_PUSH4(&array_type, &keys_vec_jl, &dims_vec_jl, &repartition_ids_vec_jl);
  const auto &dims = dist_array_->GetDims();
  auto temp_dims = dims;
  array_type = jl_apply_array_type(reinterpret_cast<jl_value_t*>(jl_int64_type), 1);
  keys_vec_jl = reinterpret_cast<jl_value_t*>(
      jl_ptr_to_array_1d(array_type, keys_.data(), keys_.size(), 0));
  dims_vec_jl = reinterpret_cast<jl_value_t*>(
      jl_ptr_to_array_1d(array_type, temp_dims.data(), temp_dims.size(), 0));

  jl_function_t *repartition_func = JuliaEvaluator::GetFunction(jl_main_module,
                                                                repartition_func_name.c_str());
  repartition_ids_vec_jl = jl_call2(repartition_func, keys_vec_jl, dims_vec_jl);
  JuliaEvaluator::AbortIfException();
  CHECK(!jl_exception_occurred()) << jl_typeof_str(jl_exception_occurred());
  int32_t *repartition_ids = reinterpret_cast<int32_t*>(jl_array_data(repartition_ids_vec_jl));

  Repartition(repartition_ids);
  JL_GC_POP();
}

void
AbstractDistArrayPartition::ComputeHashRepartitionIdsAndRepartition(
    const std::string &repartition_func_name,
    size_t num_partitions) {
  CHECK(storage_type_ == DistArrayPartitionStorageType::kKeyValueBuffer);
  jl_value_t *array_type = nullptr,
            *keys_vec_jl = nullptr,
            *dims_vec_jl = nullptr,
       *hash_vals_vec_jl = nullptr;
  JL_GC_PUSH4(&array_type, &keys_vec_jl, &dims_vec_jl, &hash_vals_vec_jl);
  const auto &dims = dist_array_->GetDims();
  auto temp_dims = dims;
  array_type = jl_apply_array_type(reinterpret_cast<jl_value_t*>(jl_int64_type), 1);
  keys_vec_jl = reinterpret_cast<jl_value_t*>(
      jl_ptr_to_array_1d(array_type, keys_.data(), keys_.size(), 0));
  dims_vec_jl = reinterpret_cast<jl_value_t*>(
      jl_ptr_to_array_1d(array_type, temp_dims.data(), temp_dims.size(), 0));

  jl_function_t *repartition_func = JuliaEvaluator::GetFunction(jl_main_module,
                                                                repartition_func_name.c_str());
  hash_vals_vec_jl = jl_call2(repartition_func, keys_vec_jl, dims_vec_jl);
  JuliaEvaluator::AbortIfException();
  CHECK(!jl_exception_occurred()) << jl_typeof_str(jl_exception_occurred());

  uint64_t *hash_vals = reinterpret_cast<uint64_t*>(jl_array_data(hash_vals_vec_jl));
  size_t num_hash_vals = jl_array_len(hash_vals_vec_jl);
  std::vector<int32_t> repartition_ids(num_hash_vals);
  for (size_t i = 0; i < num_hash_vals; i++) {
    repartition_ids[i] = static_cast<int32_t>(hash_vals[i] % num_partitions);
  }
  Repartition(repartition_ids.data());
  JL_GC_POP();
}

void
AbstractDistArrayPartition::ComputePartialModuloRepartitionIdsAndRepartition(
    size_t num_partitions,
    const std::vector<size_t> &dim_indices) {
  CHECK(storage_type_ == DistArrayPartitionStorageType::kKeyValueBuffer);
  const auto &dims = dist_array_->GetDims();
  std::vector<int32_t> repartition_ids(keys_.size());
  std::vector<int64_t> key_vec(dims.size());
  std::vector<int64_t> partial_dims(dim_indices.size());
  std::vector<int64_t> partial_key_vec(dim_indices.size());
  key::get_partial_dims(dims, dim_indices, &partial_dims);
  for (size_t i = 0; i < keys_.size(); i ++) {
    auto key = keys_[i];
    key::int64_to_vec(dims, key, &key_vec);
    key::get_partial_key(key_vec, dim_indices, &partial_key_vec);
    auto partial_key = key::vec_to_int64(partial_dims, partial_key_vec);
    int32_t repartition_id = partial_key % num_partitions;
    repartition_ids[i] = repartition_id;
  }
  Repartition(repartition_ids.data());
}

void
AbstractDistArrayPartition::ComputePartialRandomRepartitionIdsAndRepartition(
    size_t num_partitions,
    const std::vector<size_t> &dim_indices,
    std::unordered_map<int64_t, int32_t> *partial_key_to_repartition_id_map) {
  CHECK(storage_type_ == DistArrayPartitionStorageType::kKeyValueBuffer);

  const auto &dims = dist_array_->GetDims();
  std::vector<int32_t> repartition_ids(keys_.size());
  std::vector<int64_t> key_vec(dims.size());
  std::vector<int64_t> partial_dims(dim_indices.size());
  std::vector<int64_t> partial_key_vec(dim_indices.size());
  key::get_partial_dims(dims, dim_indices, &partial_dims);

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<int32_t> dist(0, num_partitions - 1);

  for (size_t i = 0; i < keys_.size(); i ++) {
    auto key = keys_[i];
    key::int64_to_vec(dims, key, &key_vec);
    key::get_partial_key(key_vec, dim_indices, &partial_key_vec);
    auto partial_key = key::vec_to_int64(partial_dims, partial_key_vec);

    auto repartition_id_iter = partial_key_to_repartition_id_map->find(partial_key);
    if (repartition_id_iter == partial_key_to_repartition_id_map->end()) {
      int32_t repartition_id = dist(gen);
      auto emplace_pair = partial_key_to_repartition_id_map->emplace(
          std::make_pair(partial_key, repartition_id));
      repartition_id_iter = emplace_pair.first;
    }
    int32_t repartition_id = repartition_id_iter->second;
    repartition_ids[i] = repartition_id;
  }
  Repartition(repartition_ids.data());
}

size_t
AbstractDistArrayPartition::GetNumUniquePartialKeys(const std::vector<size_t> &dim_indices) const {
  CHECK(storage_type_ == DistArrayPartitionStorageType::kKeyValueBuffer);
  const auto &dims = dist_array_->GetDims();
  std::vector<int64_t> key_vec(dims.size());
  std::vector<int64_t> partial_dims(dim_indices.size());
  std::vector<int64_t> partial_key_vec(dim_indices.size());
  key::get_partial_dims(dims, dim_indices, &partial_dims);
  std::set<int64_t> partial_key_set;

  for (size_t i = 0; i < keys_.size(); i ++) {
    auto key = keys_[i];
    key::int64_to_vec(dims, key, &key_vec);
    key::get_partial_key(key_vec, dim_indices, &partial_key_vec);
    auto partial_key = key::vec_to_int64(partial_dims, partial_key_vec);
    partial_key_set.emplace(partial_key);
  }
  return partial_key_set.size();
}

// remap to [remapped_partial_key_start, remapped_partial_key_end]
void
AbstractDistArrayPartition::RandomRemapPartialKeys(
    const std::vector<size_t> &dim_indices,
    std::unordered_map<int64_t, int64_t> *partial_key_to_remapped_partial_key,
    std::set<int64_t> *remapped_partial_key_set,
    int64_t remapped_partial_key_start,
    int64_t remapped_partial_key_end) {
  CHECK(storage_type_ == DistArrayPartitionStorageType::kKeyValueBuffer);
  const auto &dims = dist_array_->GetDims();
  std::vector<int64_t> key_vec(dims.size());
  std::vector<int64_t> partial_dims(dim_indices.size());
  std::vector<int64_t> partial_key_vec(dim_indices.size());
  key::get_partial_dims(dims, dim_indices, &partial_dims);

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<int64_t> dist(remapped_partial_key_start,
                                              remapped_partial_key_end);

  for (size_t i = 0; i < keys_.size(); i ++) {
    auto key = keys_[i];
    key::int64_to_vec(dims, key, &key_vec);
    key::get_partial_key(key_vec, dim_indices, &partial_key_vec);
    auto partial_key = key::vec_to_int64(partial_dims, partial_key_vec);
    auto remap_iter = partial_key_to_remapped_partial_key->find(partial_key);
    if (remap_iter == partial_key_to_remapped_partial_key->end()) {
      int64_t remapped_partial_key = dist(gen);
      size_t num_trials = 0;
      while (remapped_partial_key_set->count(remapped_partial_key) == 1) {
        num_trials += 1;
        CHECK_LT(num_trials,
                 remapped_partial_key_end - remapped_partial_key_start + 1);
        remapped_partial_key = remapped_partial_key + 1;
        if (remapped_partial_key > remapped_partial_key_end) {
          remapped_partial_key = remapped_partial_key_start;
        }
      }
      remapped_partial_key_set->emplace(remapped_partial_key);
      auto emplace_pair = partial_key_to_remapped_partial_key->emplace(
          std::make_pair(partial_key, remapped_partial_key));
      remap_iter = emplace_pair.first;
    }
    int64_t remapped_partial_key = remap_iter->second;
    key::int64_to_vec(partial_dims, remapped_partial_key, &partial_key_vec);
    key::update_key_with_partial_key(partial_key_vec, dim_indices, &key_vec);
    int64_t remapped_key = key::vec_to_int64(dims, key_vec);
    keys_[i] = remapped_key;
  }
  sorted_ = false;
}

void
AbstractDistArrayPartition::AccumHistogram(size_t dim_index,
                                           size_t num_bins,
                                           std::vector<size_t> *histogram_vec,
                                           size_t full_bin_size,
                                           size_t num_full_bins,
                                           size_t bin_size,
                                           int64_t full_bin_cutoff_key,
                                           int64_t dim_divider,
                                           int64_t dim_mod) const {
  CHECK(storage_type_ == DistArrayPartitionStorageType::kKeyValueBuffer);

  for (size_t i = 0; i < keys_.size(); i++) {
    auto key = keys_[i];
    int64_t dim_key = (key / dim_divider) % dim_mod;
    int64_t bin_key = 0;
    if (dim_key <= full_bin_cutoff_key) {
      bin_key = dim_key / full_bin_size;
    } else {
      bin_key = (dim_key - full_bin_size * num_full_bins) / bin_size + num_full_bins;
    }
    CHECK_LT(bin_key, num_bins) << " key = " << key
                                << " dim_key = " << dim_key
                                << " dim_divider = " << dim_divider
                                << " dim_mod = " << dim_mod
                                << " bin_size = " << bin_size
                                << " full_bin_size = " << full_bin_size;
    (*histogram_vec)[bin_key] += 1;
  }
}

void
AbstractDistArrayPartition::ComputePrefetchIndices(
    const std::string &prefetch_batch_func_name,
    const std::vector<int32_t> &dist_array_ids_vec,
    const std::unordered_map<int32_t, DistArray*> &global_indexed_dist_arrays,
    const std::vector<jl_value_t*> &global_read_only_var_vals,
    const std::vector<std::string> &accumulator_var_syms,
    PointQueryKeyDistArrayMap *point_key_vec_map,
    size_t offset,
    size_t num_elements) {

  CHECK(storage_type_ == DistArrayPartitionStorageType::kKeyValueBuffer);
  size_t num_args = global_read_only_var_vals.size()
                    + accumulator_var_syms.size() + 7;
  jl_value_t **jl_values;
  JL_GC_PUSHARGS(jl_values, num_args + 5);

  jl_value_t* &keys_vec_jl = jl_values[0];
  jl_value_t* &values_vec_jl = jl_values[1];
  jl_value_t* &dims_vec_jl = jl_values[2];
  jl_value_t* &ids_array_jl = jl_values[3];
  jl_value_t* &global_indexed_dist_array_dims_vec_jl = jl_values[4];
  jl_value_t* &offset_jl = jl_values[5];
  jl_value_t* &num_elements_jl = jl_values[6];

  jl_value_t* &global_indexed_dist_array_dims_jl = jl_values[num_args];
  jl_value_t* &keys_array_type_jl = jl_values[num_args + 1];
  jl_value_t* &ids_array_type_jl = jl_values[num_args + 2];
  jl_value_t* &dims_array_type_jl = jl_values[num_args + 3];
  jl_value_t* &ret_jl = jl_values[num_args + 4];

  offset_jl = jl_box_uint64(offset);
  num_elements_jl = jl_box_uint64(num_elements);
  size_t args_index = 7;
  for (size_t i = 0; i < global_read_only_var_vals.size(); i++, args_index++) {
    jl_values[args_index] = global_read_only_var_vals[i];
  }

  for (size_t i = 0; i < accumulator_var_syms.size(); i++, args_index++) {
    const auto &var_sym = accumulator_var_syms[i];
    JuliaEvaluator::GetVarJlValue(var_sym, &jl_values[args_index]);
  }
  std::vector<int32_t> temp_dist_array_ids_vec = dist_array_ids_vec;

  const auto &dims = GetDims();
  auto temp_dims = dims;
  keys_array_type_jl = jl_apply_array_type(reinterpret_cast<jl_value_t*>(jl_int64_type), 1);
  ids_array_type_jl = jl_apply_array_type(reinterpret_cast<jl_value_t*>(jl_int32_type), 1);
  dims_array_type_jl = jl_apply_array_type(keys_array_type_jl, 1);

  dims_vec_jl = reinterpret_cast<jl_value_t*>(jl_ptr_to_array_1d(
      keys_array_type_jl, temp_dims.data(), temp_dims.size(), 0));
  keys_vec_jl = reinterpret_cast<jl_value_t*>(jl_ptr_to_array_1d(
      keys_array_type_jl, keys_.data(), keys_.size(), 0));

  ids_array_jl = reinterpret_cast<jl_value_t*>(jl_ptr_to_array_1d(
      ids_array_type_jl, temp_dist_array_ids_vec.data(),
      temp_dist_array_ids_vec.size(), 0));
  global_indexed_dist_array_dims_vec_jl = reinterpret_cast<jl_value_t*>(jl_alloc_array_1d(
      dims_array_type_jl, dist_array_ids_vec.size()));
  std::vector<std::vector<int64_t>> global_indexed_dist_array_dims_array(
      dist_array_ids_vec.size());
  for (size_t i = 0; i < dist_array_ids_vec.size(); i++) {
    auto dist_array_id = dist_array_ids_vec[i];
    const auto *global_indexed_dist_array = global_indexed_dist_arrays.at(dist_array_id);
    const auto &global_indexed_dist_array_dims = global_indexed_dist_array->GetDims();
    global_indexed_dist_array_dims_array[i] = global_indexed_dist_array_dims;
    global_indexed_dist_array_dims_jl = reinterpret_cast<jl_value_t*>(jl_ptr_to_array_1d(
        keys_array_type_jl, global_indexed_dist_array_dims_array[i].data(),
        global_indexed_dist_array_dims_array[i].size(), 0));
    jl_arrayset(reinterpret_cast<jl_array_t*>(global_indexed_dist_array_dims_vec_jl),
                global_indexed_dist_array_dims_jl, i);
  }

  GetJuliaValueArray(&values_vec_jl);

  jl_function_t *prefetch_batch_func
      = JuliaEvaluator::GetFunction(jl_main_module,
                                    prefetch_batch_func_name.c_str());
  ret_jl = jl_call(prefetch_batch_func, jl_values, num_args);
  JuliaEvaluator::AbortIfException();
  jl_value_t *point_key_dist_array_vec_jl =  ret_jl;

  for (size_t i = 0; i < dist_array_ids_vec.size(); i++) {
    int32_t dist_array_id = dist_array_ids_vec[i];
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
  }
  JL_GC_POP();
}

void
AbstractDistArrayPartition::Execute(
    const std::string &loop_batch_func_name,
    const std::vector<jl_value_t*> &accessed_dist_arrays,
    const std::vector<jl_value_t*> &accessed_dist_array_buffers,
    const std::vector<jl_value_t*> &global_read_only_var_vals,
    const std::vector<std::string> &accumulator_var_syms,
    size_t offset,
    size_t num_elements) {
  LOG(INFO) << __func__ << "Start";
  CHECK(storage_type_ == DistArrayPartitionStorageType::kKeyValueBuffer);
  size_t num_args = accessed_dist_arrays.size() + accessed_dist_array_buffers.size()
                    + global_read_only_var_vals.size()
                    + accumulator_var_syms.size() + 5;
  jl_value_t **jl_values;
  JL_GC_PUSHARGS(jl_values, num_args + 3);

  jl_value_t *&keys_vec_jl = jl_values[0],
           *&values_vec_jl = jl_values[1],
             *&dims_vec_jl = jl_values[2],
              *&offset_jl = jl_values[3],
         *&num_elements_jl = jl_values[4],
      *&dims_array_type_jl = jl_values[num_args],
      *&keys_array_type_jl = jl_values[num_args + 1],
      *&accumulator_val_jl = jl_values[num_args + 2];
  offset_jl = jl_box_uint64(offset);
  num_elements_jl = jl_box_uint64(num_elements);

  size_t args_index = 5;
  for (size_t i = 0; i < accessed_dist_arrays.size(); i++, args_index++) {
    jl_values[args_index] = JuliaEvaluator::GetDistArrayAccessor(accessed_dist_arrays[i]);
  }

  for (size_t i = 0; i < accessed_dist_array_buffers.size(); i++, args_index++) {
    jl_values[args_index] = JuliaEvaluator::GetDistArrayAccessor(accessed_dist_array_buffers[i]);
  }

  for (size_t i = 0; i < global_read_only_var_vals.size(); i++, args_index++) {
    jl_values[args_index] = global_read_only_var_vals[i];
  }

  for (size_t i = 0; i < accumulator_var_syms.size(); i++, args_index++) {
    const auto &var_sym = accumulator_var_syms[i];
    JuliaEvaluator::GetVarJlValue(var_sym, &accumulator_val_jl);
    jl_values[args_index] = accumulator_val_jl;
  }

  const auto &dims = GetDims();
  auto temp_dims = dims;
  dims_array_type_jl = jl_apply_array_type(reinterpret_cast<jl_value_t*>(jl_int64_type), 1);
  keys_array_type_jl = jl_apply_array_type(reinterpret_cast<jl_value_t*>(jl_int64_type), 1);

  dims_vec_jl = reinterpret_cast<jl_value_t*>(jl_ptr_to_array_1d(
      dims_array_type_jl, temp_dims.data(), temp_dims.size(), 0));
  keys_vec_jl = reinterpret_cast<jl_value_t*>(jl_ptr_to_array_1d(
      keys_array_type_jl, keys_.data(), keys_.size(), 0));
  GetJuliaValueArray(&values_vec_jl);
  JuliaEvaluator::AbortIfException();

  jl_function_t *exec_loop_func
      = JuliaEvaluator::GetFunction(jl_main_module,
                                    loop_batch_func_name.c_str());
  LOG(INFO) << __func__ << " JLCall Start";
  jl_call(exec_loop_func, jl_values, num_args);
  JuliaEvaluator::AbortIfException();
  JL_GC_POP();
  LOG(INFO) << __func__ << "End";
}

void
AbstractDistArrayPartition::BuildIndex() {
  CHECK(storage_type_ == DistArrayPartitionStorageType::kKeyValueBuffer);
  auto &dist_array_meta = dist_array_->GetMeta();
  bool is_dense = dist_array_meta.IsDense() && dist_array_meta.IsContiguousPartitions();
  if (is_dense) {
    BuildDenseIndex();
  } else {
    BuildSparseIndex();
  }
}

void
AbstractDistArrayPartition::CheckAndBuildIndex() {
  CHECK(storage_type_ == DistArrayPartitionStorageType::kKeyValueBuffer);
  auto &dist_array_meta = dist_array_->GetMeta();
  auto index_type = dist_array_meta.GetIndexType();
  if (index_type == DistArrayIndexType::kNone) return;
  BuildIndex();
}

void
AbstractDistArrayPartition::ApplyBufferedUpdates(
    AbstractDistArrayPartition* dist_array_buffer,
    const std::vector<AbstractDistArrayPartition*> &helper_dist_arrays,
    const std::vector<AbstractDistArrayPartition*> &helper_dist_array_buffers,
    const std::string &apply_buffer_func_name) {
  size_t num_helper_dist_arrays = helper_dist_arrays.size();
  size_t num_helper_dist_array_buffers = helper_dist_array_buffers.size();
  size_t num_args = num_helper_dist_arrays + num_helper_dist_array_buffers * 2 + 3;
  jl_value_t **args_vec;
  JL_GC_PUSHARGS(args_vec, num_args + 1);

  jl_value_t *&buffered_updates_keys_jl = args_vec[0],
           *&buffered_updates_values_jl = args_vec[1],
                *&dist_array_values_jl = args_vec[2],
                    *&key_array_type_jl = args_vec[num_args];

  key_array_type_jl = jl_apply_array_type(reinterpret_cast<jl_value_t*>(jl_int64_type), 1);
  auto &dist_array_buffer_keys = dist_array_buffer->GetKeys();
  buffered_updates_keys_jl = reinterpret_cast<jl_value_t*>(
      jl_ptr_to_array_1d(key_array_type_jl,
                         dist_array_buffer_keys.data(),
                         dist_array_buffer_keys.size(), 0));
  dist_array_buffer->GetJuliaValueArray(&buffered_updates_values_jl);
  GetJuliaValueArray(dist_array_buffer_keys, &dist_array_values_jl);

  size_t args_index = 3;
  for (auto *helper_dist_array_partition : helper_dist_arrays) {
    helper_dist_array_partition->GetJuliaValueArray(dist_array_buffer_keys,
                                                    &args_vec[args_index]);
    args_index++;
  }

  for (auto *helper_dist_array_buffer_partition : helper_dist_array_buffers) {
    auto &helper_dist_array_buffer_partition_keys = helper_dist_array_buffer_partition->GetKeys();
    args_vec[args_index] = reinterpret_cast<jl_value_t*>(
        jl_ptr_to_array_1d(key_array_type_jl,
                           helper_dist_array_buffer_partition_keys.data(),
                           helper_dist_array_buffer_partition_keys.size(), 0));
    helper_dist_array_buffer_partition->GetJuliaValueArray(&args_vec[args_index + 1]);
    args_index += 2;
  }

  jl_function_t *apply_buffer_func = JuliaEvaluator::GetFunction(
      jl_main_module, apply_buffer_func_name.c_str());
  jl_call(apply_buffer_func, args_vec, num_args);
  JuliaEvaluator::AbortIfException();
  SetJuliaValues(dist_array_buffer_keys, dist_array_values_jl);

  args_index = 3;
  for (auto *helper_dist_array_partition : helper_dist_arrays) {
    helper_dist_array_partition->SetJuliaValues(dist_array_buffer_keys,
                                                args_vec[args_index]);
    args_index++;
  }
  JL_GC_POP();
  JuliaEvaluator::AbortIfException();
}

}
}
