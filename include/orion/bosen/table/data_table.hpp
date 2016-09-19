#pragma once

#include <string>
#include <list>
#include <stdint.h>
#include <orion/constants.hpp>
#include <orion/bosen/table.hpp>
#include <orion/helper.hpp>
#include <stdio.h>
#include <algorithm>
#include <unordered_map>
#include <vector>
#include <orion/noncopyable.hpp>
#include <orion/bosen/rating_partition.hpp>

namespace orion {
namespace bosen {
class DataTable {
 private:
  const std::string data_path_;
  const size_t num_partitions_;
  const int32_t my_id_;
  static constexpr size_t kDataBufferCapacity = 1 * k1_Mi;
  std::unique_ptr<Rating[]> data_buff_;
  size_t num_values_ {0};
  int32_t max_x_ {0}, max_y_ {0};
  std::unordered_map<int32_t, RatingBuffer> partition_out_;
  std::list<RatingBuffer> partition_in_;
  std::unordered_map<int32_t, RatingBuffer> partition_my_;
  size_t num_partitions_recved_ {0};
  std::vector<RatingPartition> data_partitions_;
  size_t num_local_partitions_ {0};
  size_t local_x_min_ {0};
  size_t num_rows_per_partition_ {0};
  size_t num_cols_per_partition_ {0};
  size_t data_partition_begin_ = {0};
  size_t data_partition_iter_ = {0};
 public:
  DataTable (const char *data_path,
              size_t num_partitions,
              int32_t my_id):
      data_path_(data_path),
      num_partitions_(num_partitions),
      my_id_(my_id) { }
  ~DataTable() { }

  void Init();
  void GetMaxXY(int32_t *x, int32_t *y);
  void set_max_xy(int32_t x, int32_t y) {
    max_x_ = x;
    max_y_ = y;
  }
  void PrepareRangePartition();
  const std::unordered_map<int32_t, RatingBuffer> &
  get_partition_out() { return partition_out_; }

  void inc_num_partitions_recved() {
    num_partitions_recved_++;
  }

  bool recved_all_partitions() {
    return num_partitions_recved_ == num_partitions_;
  }

  void AddRecvedRatings(int32_t executor_id, size_t buffer_capacity,
                         size_t offset, const uint8_t *recv_mem, size_t received_size) {
    auto buffer_iter = partition_my_.find(executor_id);
    if (buffer_iter == partition_my_.end()) {
      partition_my_.emplace(executor_id,
                            std::make_pair(std::make_unique<Rating[]>(buffer_capacity),
                                           buffer_capacity));
      buffer_iter = partition_my_.find(executor_id);
    }
    uint8_t *ratings_mem = reinterpret_cast<uint8_t*>(buffer_iter->second.first.get()) + offset;
    memcpy(ratings_mem, recv_mem, received_size);
  }

  void ProcessMyPartition() {
    if (partition_in_.empty()) return;
    auto &buffer = *(partition_in_.begin());
    partition_my_.emplace(my_id_,
                          std::move(buffer));
  }

  void FinalizeRangePartition() {
    size_t total_size = 0;
    for (auto buff_iter = partition_my_.begin(); buff_iter != partition_my_.end();
         buff_iter++) {
      total_size += buff_iter->second.second;
    }
    CHECK_GT(total_size, 0);
    data_buff_ = std::make_unique<Rating[]>(total_size);
    size_t offset = 0;
    for (auto buff_iter = partition_my_.begin(); buff_iter != partition_my_.end();
         buff_iter++) {
      memcpy(reinterpret_cast<uint8_t*>(data_buff_.get()) + offset,
             reinterpret_cast<uint8_t*>(buff_iter->second.first.get()),
             buff_iter->second.second * sizeof(Rating));
      offset += buff_iter->second.second * sizeof(Rating);
    }
    num_values_ = total_size;
    std::sort(data_buff_.get(), data_buff_.get() + num_values_,
              [](const Rating& a, const Rating& b) -> bool {
                if (a.x < b.x) return true;
                else if (a.x == b.x && a.y <= b.y) return true;
                else return false;
              });
    partition_my_.clear();
    partition_in_.clear();
  }

  void clear_partition_out() {
    partition_out_.clear();
  }

  void PartitionRatings(size_t pipeline_depth);

  void BeginIterating() {
    data_partition_iter_ = data_partition_begin_;
  }

  RatingPartition &NextPartition() {
    auto &partition = data_partitions_[data_partition_iter_];
    data_partition_iter_++;
    data_partition_iter_ %= data_partitions_.size();
    return partition;
  }

  const std::vector<RatingPartition> &get_data_partitions() {
    return data_partitions_;
  }

  size_t get_data_partition_local_begin() {
    return data_partition_begin_;
  }

  size_t get_num_rows_per_partition() {
    return num_rows_per_partition_;
  }

  size_t get_num_cols_per_partition() {
    return num_cols_per_partition_;
  }

  size_t get_local_x_min() {
    return local_x_min_;
  }
};

void DataTable::Init() {
  if (data_path_.empty()) return;

  FILE *data_file = fopen(data_path_.c_str(), "r");
  CHECK(data_file) << data_path_ << " open failed";
  fseek(data_file, 0, SEEK_END);
  size_t file_size = ftell(data_file);
  size_t file_partition_size
      = (file_size + num_partitions_ - 1) / num_partitions_;
  size_t num_actual_partitions = (file_size + file_partition_size - 1) / file_partition_size;
  size_t last_partition_size = file_size - (num_actual_partitions - 1) * file_partition_size;
  if (num_actual_partitions != num_partitions_)
    LOG(WARNING) << "only " << num_actual_partitions << " partitions to read!";
  //CHECK(file_partition_size * (num_partitions_ - 1) < file_size)
  //    << "file is too small for this many partitions";
  size_t file_read_size
    = (my_id_ >= num_actual_partitions)
    ? 0
    : ((my_id_ == num_actual_partitions - 1)
       ? last_partition_size
       : ((num_actual_partitions >= 2 && my_id_ == num_actual_partitions - 2)
	  ? file_size - file_partition_size * my_id_
	  : file_partition_size * 2));
  if (file_read_size == 0) return;

  std::unique_ptr<char[]> text_data = std::make_unique<char[]>(file_read_size + 1);

  fseek(data_file, file_partition_size * my_id_, SEEK_SET);
  size_t read_count = fread(text_data.get(), file_read_size, 1, data_file);
  CHECK_EQ(read_count, 1) << "ferror = " << ferror(data_file)
                          << " feof = " << feof(data_file)
                          << " file_read_size = " << file_read_size
                          << " file_size = " << file_size;
  LOG(INFO) << "data read done";
  fclose(data_file);

  size_t buffer_start_offset = 0, buffer_end_offset = file_partition_size;
  if (my_id_ > 0) {
    for (; buffer_start_offset < file_read_size; buffer_start_offset++) {
      if (text_data.get()[buffer_start_offset] == '\n') {
        buffer_start_offset++;
        break;
      }
    }
  }

  if (my_id_ != num_partitions_ - 1) {
    for (; buffer_end_offset < file_read_size; buffer_end_offset++) {
      if (text_data.get()[buffer_end_offset] == '\n') break;
    }
    CHECK_LT (buffer_end_offset, file_read_size) << "cannot handle this case yet";
  } else {
    buffer_end_offset = file_read_size;
  }

  if (buffer_start_offset >= buffer_end_offset) return;

  text_data.get()[buffer_end_offset] = '\0';

  std::list<RatingBuffer> data_buff;
  size_t num_vals_per_buff = kDataBufferCapacity / sizeof(Rating);
  data_buff.emplace_back(std::make_unique<Rating[]>(num_vals_per_buff), 0);
  auto buff_iter = data_buff.begin();
  Rating *curr_buff = buff_iter->first.get();
  size_t *curr_buff_size = &buff_iter->second;

  char *next_pos = text_data.get() + buffer_start_offset;
  while (next_pos - text_data.get() < buffer_end_offset) {
    int32_t xid = strtol(next_pos, &next_pos, 10);
    CHECK(*next_pos == ',') << "xid = " << xid << " actucally " << *next_pos;
    int32_t yid = strtol(next_pos + 1, &next_pos, 10);
    CHECK(*next_pos == ',') << "xid = " << xid << " yid = " << yid << " actually " << *next_pos;
    float rating = strtof(next_pos + 1, &next_pos);
    CHECK(*next_pos == '\n' || *next_pos == '\0') << *next_pos;
    next_pos++;
    if (*curr_buff_size >= num_vals_per_buff) {
      data_buff.emplace_back(std::make_unique<Rating[]>(num_vals_per_buff), 0);
      num_values_ += *curr_buff_size;
      buff_iter++;
      curr_buff = buff_iter->first.get();
      curr_buff_size = &buff_iter->second;
    }
    curr_buff[*curr_buff_size].x = xid;
    curr_buff[*curr_buff_size].y = yid;
    curr_buff[*curr_buff_size].v = rating;
    (*curr_buff_size)++;
  };
  num_values_ += *curr_buff_size;

  data_buff_ = std::make_unique<Rating[]>(num_values_);
  size_t offset = 0;
  for (buff_iter = data_buff.begin(); buff_iter != data_buff.end();
       buff_iter++) {
    uint8_t *buff = reinterpret_cast<uint8_t*>(buff_iter->first.get());
    size_t buff_size = buff_iter->second;
    memcpy(reinterpret_cast<uint8_t*>(data_buff_.get()) + offset, buff,
           buff_size * sizeof(Rating));
    offset += buff_size * sizeof(Rating);
  }
}

void
DataTable::GetMaxXY(int32_t *x, int32_t *y) {
  int32_t max_x = 0, max_y = 0;
  for (size_t i = 0; i < num_values_; ++i) {
    Rating &r = data_buff_.get()[i];
    if (r.x > max_x) max_x = r.x;
    if (r.y > max_y) max_y = r.y;
  }
  *x = max_x;
  *y = max_y;
}

void
DataTable::PrepareRangePartition() {
  std::unordered_map<int32_t,
                     std::list<RatingBuffer>> partitioned_ratings;
  size_t num_rows_per_partition = (max_x_ + num_partitions_) / num_partitions_;
  size_t num_vals_per_buff = kDataBufferCapacity / sizeof(Rating);
  local_x_min_ = num_rows_per_partition * my_id_;
  num_rows_per_partition_ = num_rows_per_partition;
  // put values to partitions
  for (size_t i = 0; i < num_values_; ++i) {
    Rating &r = data_buff_.get()[i];
    int32_t partition_id = r.x / num_rows_per_partition;
    auto partition_iter = partitioned_ratings.find(partition_id);
    if (partition_iter == partitioned_ratings.end()) {
      partitioned_ratings.emplace(partition_id,
                                  std::list<RatingBuffer>());
      partition_iter = partitioned_ratings.find(partition_id);
      partition_iter->second.emplace_front(
          std::make_unique<Rating[]>(num_vals_per_buff),
          0);
    }
    auto buff_iter = partition_iter->second.begin();
    if (buff_iter->second == num_vals_per_buff) {
      partition_iter->second.emplace_front(
          std::make_unique<Rating[]>(num_vals_per_buff),
          0);
      buff_iter = partition_iter->second.begin();
    }
    Rating *buff = buff_iter->first.get();
    size_t &buff_size = buff_iter->second;
    buff[buff_size] = r;
    buff_size++;
    CHECK_LE(r.y, max_y_);
  }

  for (int32_t partition_id = 0; partition_id < num_partitions_; ++partition_id) {
    auto partition_iter = partitioned_ratings.find(partition_id);
    if (partition_iter == partitioned_ratings.end()) {
      if (partition_id != my_id_)
        partition_out_.emplace(partition_id, std::make_pair(nullptr, 0));
    } else {
      auto &buffer_list = partition_iter->second;
      size_t num_buffers = buffer_list.size();
      CHECK(num_buffers >= 1);
      size_t num_vals_this_partition = (num_buffers - 1) * num_vals_per_buff;
      auto buffer_iter = buffer_list.begin();
      num_vals_this_partition += buffer_iter->second;
      if (partition_id == my_id_) {
        partition_in_.emplace_front(
            std::make_unique<Rating[]>(num_vals_this_partition),
            num_vals_this_partition);
        Rating *dest_buff = partition_in_.begin()->first.get();
        size_t offset = 0;
        for (auto buffer_iter = buffer_list.begin(); buffer_iter != buffer_list.end();
             buffer_iter++) {
          memcpy(reinterpret_cast<uint8_t*>(dest_buff) + offset,
                 reinterpret_cast<uint8_t*>(buffer_iter->first.get()),
                 sizeof(Rating) * buffer_iter->second);
          offset += buffer_iter->second * sizeof(Rating);
        }
      } else {
        auto partition_iter = partition_out_.find(partition_id);
        if (partition_iter == partition_out_.end()) {
          partition_out_.emplace(partition_id,
                                 std::make_pair(std::make_unique<Rating[]>(num_vals_this_partition),
                                                num_vals_this_partition));
          partition_iter = partition_out_.find(partition_id);
        }
        Rating *dest_buff = partition_iter->second.first.get();
        size_t offset = 0;
        for (auto buffer_iter = buffer_list.begin(); buffer_iter != buffer_list.end();
             buffer_iter++) {
          memcpy(reinterpret_cast<uint8_t*>(dest_buff) + offset,
                 reinterpret_cast<uint8_t*>(buffer_iter->first.get()),
                 sizeof(Rating) * buffer_iter->second);
          offset += buffer_iter->second * sizeof(Rating);
        }
      }
    }
  }
}

void
DataTable::PartitionRatings(size_t pipeline_depth) {
  num_local_partitions_ = num_partitions_ * pipeline_depth;
  data_partitions_.resize(num_local_partitions_);
  size_t num_ys_per_partition
      = (max_y_ + num_local_partitions_) / num_local_partitions_;
  num_cols_per_partition_ = num_ys_per_partition;
  Rating *ratings = data_buff_.get();
  for (size_t i = 0; i < num_values_; i++) {
    size_t partition = ratings[i].y / num_ys_per_partition;
    CHECK_LE(partition, num_local_partitions_ - 1) << "ratings.y = " << ratings[i].y
                                                   << " i = " << i
                                                   << " max_y_ = " << max_y_;
    data_partitions_[partition].num_ratings++;
  }
  size_t partition_id = 0;
  for (auto & partition : data_partitions_) {
    //LOG(INFO) << "partition_id = " << partition_id
    //	      << " num_ratings = " << partition.num_ratings;
    partition.ratings = std::make_unique<Rating[]>(partition.num_ratings);
    partition.y_min = num_ys_per_partition * partition_id;
    partition.y_max = num_ys_per_partition * (partition_id + 1);
    partition_id++;
  }

  std::vector<size_t> buff_size(num_local_partitions_, 0);
  for (size_t i = 0; i < num_values_; i++) {
    size_t partition = ratings[i].y / num_ys_per_partition;
    //CHECK_LT(buff_size[partition], data_partitions_[partition].num_ratings);
    data_partitions_[partition].ratings.get()[buff_size[partition]] = ratings[i];
    buff_size[partition]++;
  }
  data_partition_begin_ = pipeline_depth * my_id_;
}

}
}
