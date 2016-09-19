#include <glog/logging.h>
#include <orion/perf.hpp>

size_t num_bytes = 8000*1000;

int *mem;

std::vector<orion::PerfCount::CountType> vec(6);


int main(int argc __attribute__((unused)), char *argv[]) {
  google::InitGoogleLogging(argv[0]);
  LOG(INFO) << "hello!";
  vec[0] = orion::PerfCount::PERF_COUNT_TYPE_HW_CPU_CYCLES;
  vec[1] = orion::PerfCount::PERF_COUNT_TYPE_HW_INSTRUCTIONS;
  vec[2] = orion::PerfCount::PERF_COUNT_TYPE_HW_CACHE_REFERENCES;
  vec[3] = orion::PerfCount::PERF_COUNT_TYPE_HW_CACHE_MISSES;
  vec[4] = orion::PerfCount::PERF_COUNT_TYPE_HW_CACHE_L1D_READ_ACCESS;
  vec[5] = orion::PerfCount::PERF_COUNT_TYPE_HW_CACHE_L1D_WRITE_ACCESS;
  mem = new int[num_bytes];
  orion::PerfCount perf_count(vec);
  perf_count.Start();
  int sum = 0;
  for (uint j = 0; j < 128; ++j) {
    for (uint i = j; i < num_bytes; i += 128) {
      mem[i] = 1;
    }

    for (uint i = j; i < num_bytes; i += 128) {
      mem[i] += 1;
    }

    for (uint i = j; i < num_bytes; i += 128) {
      sum += mem[i];
    }
  }
  LOG(INFO) << "sum = " << sum;
  perf_count.Stop();

  LOG(INFO) << "CPU_CYCLES = "
            << perf_count.GetByType(orion::PerfCount::PERF_COUNT_TYPE_HW_CPU_CYCLES)
            << " HW_INSTRUCTIONS = "
            << perf_count.GetByType(orion::PerfCount::PERF_COUNT_TYPE_HW_INSTRUCTIONS)
            << " HW_CACHE_REFERENCES = "
            << perf_count.GetByType(
                orion::PerfCount::PERF_COUNT_TYPE_HW_CACHE_REFERENCES)
            << " HW_CACHE_MISSES = "
            << perf_count.GetByType(
                orion::PerfCount::PERF_COUNT_TYPE_HW_CACHE_MISSES)
            << " L1D_READ_ACCESS = "
            << perf_count.GetByType(
                orion::PerfCount::PERF_COUNT_TYPE_HW_CACHE_L1D_READ_ACCESS)
            << " L1D_WRITE_ACCESS = "
            << perf_count.GetByType(
                orion::PerfCount::PERF_COUNT_TYPE_HW_CACHE_L1D_WRITE_ACCESS);
  return 0;
}
