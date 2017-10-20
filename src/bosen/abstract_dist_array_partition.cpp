#include <orion/bosen/abstract_dist_array_partition.hpp>
#include <glog/logging.h>

namespace orion {
namespace bosen {

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
    std::vector<char> *char_buff, size_t *begin,
    size_t *end) {
#ifdef ORION_USE_HDFS
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

  char_buff->reserve(read_size + 1);
  hdfsFile data_file = hdfsOpenFile(fs, file_path.c_str(),
                                    O_RDONLY, 0, 0, 0);
  CHECK(data_file != NULL);

  size_t total_read_size = 0;
  const size_t kMaxReadSize = 64 * 1024 * 1024;
  while (total_read_size < read_size) {
    size_t curr_read_size = read_size - total_read_size;
    curr_read_size = (curr_read_size > kMaxReadSize) ? kMaxReadSize : curr_read_size;
    tSize read_count = hdfsPread(fs, data_file,
                                 read_offset + total_read_size,
                                 char_buff->data() + total_read_size,
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
                       char_buff, begin, end);
  (*char_buff)[*end] = '\0';
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
    std::vector<char> *char_buff, size_t *begin, size_t *end) {
  FILE *data_file = fopen(file_path.c_str(), "r");
  CHECK(data_file) << file_path << " open failed";
  fseek(data_file, 0, SEEK_END);
  size_t file_size = ftell(data_file);
  size_t read_offset = partition_id * partition_size;

  if (read_offset >= file_size) return false;
  size_t read_size = std::min(read_offset + partition_size * 2,
                              file_size - read_offset);

  fseek(data_file, read_offset, SEEK_SET);
  char_buff->reserve(read_size + 1);
  size_t read_count = fread(char_buff->data(),
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
                       char_buff, begin, end);
  (*char_buff)[*end] = '\0';
  return true;
}

}
}
