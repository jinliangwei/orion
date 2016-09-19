#include <glog/logging.h>
#include <stdio.h>
#include <memory>
#include <random>
#include <algorithm>
#include <fstream>
#include <orion/helper.hpp>

int32_t max_x {0}, max_y {0};

int
main(int argc, char *argv[]) {
  //  const char *data_path = "/home/jinliang/mf/ml-10M100K/ratings.csv";
  //  const char *output_path = "/home/jinliang/mf/ml-10M100K/ratings_shuffled.csv";
  //const char *data_path = "/home/jinliang/mf/ml-latest/ratings_p.csv";
  //const char *output_path = "/home/jinliang/mf/ml-latest/ratings_shuffled.csv";
  const char *data_path = "/proj/BigLearning/jinlianw/data/matrixfact_data/netflix.csv";
  const char *output_path = "/proj/BigLearning/jinlianw/data/matrixfact_data/netflix.shuffled.csv";

  FILE *data_file = fopen(data_path, "r");
  CHECK(data_file) << data_path << " open failed";
  fseek(data_file, 0, SEEK_END);
  size_t file_size = ftell(data_file);
  fseek(data_file, 0, SEEK_SET);
  std::unique_ptr<char[]> text_buff = std::make_unique<char[]>(file_size + 1);
  size_t read_count = fread(text_buff.get(), file_size, 1, data_file);
  CHECK_EQ(read_count, 1);
  text_buff.get()[file_size] = '\0';
  fclose(data_file);

  char *next_pos = text_buff.get();
  while (next_pos - text_buff.get() < file_size) {
    int32_t xid = strtol(next_pos, &next_pos, 10);
    CHECK(*next_pos == ',') << "xid = " << xid << " actucally " << *next_pos;
    int32_t yid = strtol(next_pos + 1, &next_pos, 10);
    CHECK(*next_pos == ',') << "xid = " << xid << " yid = " << yid << " actually " << *next_pos;
    strtof(next_pos + 1, &next_pos);
    CHECK(*next_pos == '\n' || *next_pos == '\0') << *next_pos;
    next_pos++;
    if (xid > max_x) max_x = xid;
    if (yid > max_y) max_y = yid;
  }

  std::vector<int32_t> xids(max_x + 1), yids(max_y + 1);
  for (int32_t i = 0; i <= max_x; ++i) {
    xids[i] = i;
  }
  for (int32_t i = 0; i <= max_y; ++i) {
    yids[i] = i;
  }

  std::random_device rd;
  std::mt19937 g(rd());

  std::shuffle(xids.begin(), xids.end(), g);
  std::shuffle(yids.begin(), yids.end(), g);

  std::ofstream ofs(output_path, std::ios::out | std::ios::trunc);
  next_pos = text_buff.get();
  while (next_pos - text_buff.get() < file_size) {
    int32_t xid = strtol(next_pos, &next_pos, 10);
    CHECK(*next_pos == ',') << "xid = " << xid << " actucally " << *next_pos;
    int32_t yid = strtol(next_pos + 1, &next_pos, 10);
    CHECK(*next_pos == ',') << "xid = " << xid << " yid = " << yid << " actually " << *next_pos;
    float rating = strtof(next_pos + 1, &next_pos);
    CHECK(*next_pos == '\n' || *next_pos == '\0') << *next_pos;
    next_pos++;
    ofs << xids[xid] << "," << yids[yid] << "," << rating << "\n";
  }
  ofs.close();

  return 0;
}
