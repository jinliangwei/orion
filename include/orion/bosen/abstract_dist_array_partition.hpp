#pragma once

#include <stdint.h>
#include <string>
#include <hdfs.h>
#include <orion/bosen/blob.hpp>
#include <orion/bosen/julia_module.hpp>

namespace orion {
namespace bosen {

class AbstractDistArrayPartition {
 public:
  AbstractDistArrayPartition() { }
  virtual ~AbstractDistArrayPartition() { }

  virtual bool LoadTextFile(
      JuliaEvaluator *julia_eval,
      const std::string &file_path, int32_t partition_id,
      bool flatten_results, bool value_only, bool parse,
      size_t num_dims,
      JuliaModule parser_func_module,
      const std::string &parser_func) = 0;

  virtual void Insert(int64_t key, const Blob &buff) = 0;
  virtual void Get(int64_t key, Blob *buff) = 0;
  virtual void GetRange(int64_t start, int64_t end, Blob *buff) = 0;
 protected:
  DISALLOW_COPY(AbstractDistArrayPartition);
  static void GetReadSizeAndOffset(size_t file_size, size_t num_partitions,
                                   size_t min_partition_size,
                                   size_t partition_id,
                                   size_t *curr_partition_size,
                                   size_t *next_partition_size,
                                   size_t *read_offset);

  static void GetBufferBeginAndEnd(size_t num_partitions,
                                   int32_t partition_id,
                                   size_t curr_partition_size,
                                   size_t next_partition_size,
                                   std::vector<char> *char_buff,
                                   size_t *begin,
                                   size_t *end);

  static bool LoadFromHDFS(const std::string &hdfs_name_node,
                           const std::string &file_path, int32_t partition_id,
                           size_t num_partitions, size_t min_partition_size,
                           std::vector<char> *char_buff, size_t *begin,
                           size_t *end);

  static bool LoadFromPosixFS(const std::string &file_path, int32_t partition_id,
                              size_t num_partitions, size_t min_partition_size,
                              std::vector<char> *char_buff, size_t *begin,
                              size_t *end);
};

void
AbstractDistArrayPartition::GetReadSizeAndOffset(
    size_t file_size, size_t num_partitions,
    size_t min_partition_size,
    size_t partition_id,
    size_t *curr_partition_size,
    size_t *next_partition_size,
    size_t *read_offset) {
  size_t partition_size = (file_size + num_partitions - 1) / num_partitions;
  LOG(INFO) << "partition_size = " << partition_size;

  if (partition_size - 1 < min_partition_size) {
    LOG(INFO) << "partition size below threshold";
    partition_size = min_partition_size;
    size_t num_partitions_to_read
        = (file_size + partition_size - 1) / partition_size;

    if (num_partitions_to_read == 1) {
      *curr_partition_size = partition_size;
      *next_partition_size = 0;
      *read_offset = 0;
      return;
    }

    size_t last_partition_size
        = file_size - partition_size * (num_partitions_to_read - 1);

    if (partition_id < num_partitions_to_read - 1) {
      *curr_partition_size = partition_size;
      *next_partition_size
          = (partition_id == num_partitions_to_read - 2) ?
          last_partition_size : partition_size;
      *read_offset = partition_id * partition_size;
    } else if(partition_id == num_partitions_to_read - 1) {
      *curr_partition_size = last_partition_size;
      *next_partition_size = 0;
      *read_offset = partition_id * partition_size;
    } else {
      *curr_partition_size = 0;
      *next_partition_size = 0;
      *read_offset = 0;
    }
  } else {
    LOG(INFO) << "partition size above threshold";
    size_t num_full_partitions
        = file_size % num_partitions == 0
        ? num_partitions : (file_size % num_partitions);
    size_t unfull_partition_size = partition_size - 1;
    LOG(INFO) << "num full partitions = " << num_full_partitions;
    if (num_partitions == 1) {
      *curr_partition_size = file_size;
      *next_partition_size = 0;
      *read_offset = 0;
      return;
    }

    if (partition_id < num_full_partitions) {
      *curr_partition_size = partition_size;
      *next_partition_size
          = (partition_id < num_full_partitions - 1)
          ? partition_size
          : ((num_partitions == num_full_partitions)
             ? 0
             : unfull_partition_size);
      *read_offset = partition_id * partition_size;
      LOG(INFO) << "curr_partition_size = " << *curr_partition_size
                << " next_partition_size = " << *next_partition_size
                << " read_offset = " << *read_offset;
    } else {
      *curr_partition_size = unfull_partition_size;
      *next_partition_size
          = (partition_id == num_partitions - 1)
          ? 0
          : unfull_partition_size;
      *read_offset = num_full_partitions * partition_size
                     + (partition_id - num_full_partitions) * unfull_partition_size;
    }
  }
}

void
AbstractDistArrayPartition::GetBufferBeginAndEnd(
    size_t num_partitions,
    int32_t partition_id,
    size_t curr_partition_size,
    size_t next_partition_size,
    std::vector<char> *char_buff,
    size_t *begin,
    size_t *end) {
  *begin = 0;
  *end = curr_partition_size;
  size_t i = 0;
  if (partition_id != 0) {
    for (i = 0; i < curr_partition_size; i++) {
      if ((*char_buff)[i] == '\n') {
        *begin = i + 1;
        break;
      }
    }
    CHECK_LE(i, curr_partition_size);
  }
  if (partition_id != num_partitions - 1) {
    for (i = curr_partition_size; i < curr_partition_size + next_partition_size; i++) {
      if ((*char_buff)[i] == '\n') {
        *end = i;
        break;
      }
    }
    CHECK_LE(i, curr_partition_size + next_partition_size);
  }
}

bool
AbstractDistArrayPartition::LoadFromHDFS(
    const std::string &hdfs_name_node,
    const std::string &file_path, int32_t partition_id,
    size_t num_partitions, size_t min_partition_size,
    std::vector<char> *char_buff, size_t *begin,
    size_t *end) {
#ifdef ORION_USE_HDFS
  hdfsFS fs = hdfsConnect(hdfs_name_node.c_str(), 0);
  CHECK(fs != NULL) << hdfs_name_node;
  hdfsFileInfo* data_info = hdfsGetPathInfo(fs, file_path.c_str());
  CHECK(data_info != NULL);
  size_t file_size = data_info->mSize;
  hdfsFreeFileInfo(data_info, 1);
  size_t curr_partition_size = 0, next_partition_size = 0, read_offset = 0;
  GetReadSizeAndOffset(
      file_size, num_partitions,
      min_partition_size, partition_id,
      &curr_partition_size, &next_partition_size,
      &read_offset);

  if (curr_partition_size == 0) return false;
  size_t file_read_size = curr_partition_size + next_partition_size;
  char_buff->reserve(curr_partition_size + next_partition_size + 1);
  hdfsFile data_file = hdfsOpenFile(fs, file_path.c_str(),
                                    O_RDONLY, 0, 0, 0);
  CHECK(data_file != NULL);

  size_t total_read_size = 0;
  const size_t kMaxReadSize = 64 * 1024 * 1024;
  while (total_read_size < file_read_size) {
    size_t read_size = file_read_size - total_read_size;
    read_size = (read_size > kMaxReadSize) ? kMaxReadSize : read_size;
    tSize read_count = hdfsPread(fs, data_file,
                                 read_offset + total_read_size,
                                 char_buff->data() + total_read_size,
                                 read_size);
    CHECK_GT(read_count, 0) << "read error! errno = " << errno;
    total_read_size += read_count;
  }
  CHECK_EQ(total_read_size, file_read_size)
      << " file_read_size = " << file_read_size
      << " file_size = " << file_size;
  hdfsCloseFile(fs, data_file);
  GetBufferBeginAndEnd(num_partitions, partition_id, curr_partition_size,
                       next_partition_size, char_buff, begin, end);
  (*char_buff)[*end + 1] = '\0';
  return true;
#else
  LOG(FATAL) << "HDFS is not supported in this build";
  return false;
#endif
}

bool
AbstractDistArrayPartition::LoadFromPosixFS(
    const std::string &file_path, int32_t partition_id,
    size_t num_partitions, size_t min_partition_size,
    std::vector<char> *char_buff, size_t *begin, size_t *end) {
  LOG(INFO) << __func__;
  FILE *data_file = fopen(file_path.c_str(), "r");
  CHECK(data_file) << file_path << " open failed";
  fseek(data_file, 0, SEEK_END);
  size_t file_size = ftell(data_file);
  size_t curr_partition_size = 0, next_partition_size = 0, read_offset = 0;
  GetReadSizeAndOffset(
      file_size, num_partitions,
      min_partition_size, partition_id,
      &curr_partition_size, &next_partition_size,
      &read_offset);

  LOG(INFO) << "partition id = " << partition_id
            << " file size = " << file_size
            << " curr_partition size = " << curr_partition_size
            << " next_partition size = " << next_partition_size
            << " read offset = " << read_offset;

  if (curr_partition_size == 0) return false;
  fseek(data_file, read_offset, SEEK_SET);
  char_buff->reserve(curr_partition_size + next_partition_size + 1);
  size_t read_count = fread(char_buff->data(),
                            curr_partition_size + next_partition_size, 1,
                            data_file);

  CHECK_EQ(read_count, 1) << "ferror = " << ferror(data_file)
                          << " feof = " << feof(data_file)
                          << " file_read_size = " << curr_partition_size + next_partition_size
                          << " file_size = " << file_size;
  LOG(INFO) << "data read done";
  fclose(data_file);
  GetBufferBeginAndEnd(num_partitions, partition_id, curr_partition_size,
                       next_partition_size, char_buff, begin, end);
  (*char_buff)[*end + 1] = '\0';
  return true;
}

}
}
