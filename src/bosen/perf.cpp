
#include <orion/bosen/perf.hpp>
#include <sys/types.h>
#include <sys/syscall.h>
#include <glog/logging.h>
#include <errno.h>

namespace orion {
namespace bosen {

namespace {

const char *perf_count_name[]  __attribute__((unused)) =
{
  "CPUCycles",
  "Instructions",
  "CacheReferences",
  "CacheMisses",
  "BranchInstructions",
  "BranchMisses",
  "BUSCycles",
  "L1IReadAccess",
  "L1IReadMiss",
  "L1IWriteAccess",
  "L1IWriteMiss",
  "L1IPrefetchAccess",
  "L1IPrefetchMiss",
  "L1DReadAccess",
  "L1DReadMiss",
  "L1DWriteAccess",
  "L1DWriteMiss",
  "L1DPrefetchAccess",
  "L1DPrefetchMiss",
  "LLReadAccess",
  "LLReadMiss",
  "LLWriteAccess",
  "LLWriteMiss",
  "LLPrefetchAccess",
  "LLPrefetchMiss",
  "ITLBReadAccess",
  "ITLBReadMiss",
  "ITLBWriteAccess",
  "ITLBWriteMiss",
  "ITLBPrefetchAccess",
  "ITLBPrefetchMiss",
  "DTLBReadAccess",
  "DTLBReadMiss",
  "DTLBWriteAccess",
  "DTLBWriteMiss",
  "DTLBPrefetchAccess",
  "DTLBPrefetchMiss",
  "CPUClock",
  "TaskClock",
  "PageFaults",
  "ContextSwitches",
  "CPUMigrations",
  "PageFaultsMinor",
  "PageFaultsMajor",
  "AlignmentFaults",
  "EmulationFaults",
};
int sys_perf_event_open (
    perf_event_attr *attr, pid_t pid, int cpu, int group_fd,
    unsigned long flags) __attribute__((unused));

int
sys_perf_event_open (
    perf_event_attr *attr, pid_t pid, int cpu, int group_fd,
    unsigned long flags) {
  attr->size = sizeof(*attr);
  return (int)syscall(__NR_perf_event_open, attr, pid, cpu, group_fd,
                      flags);
}

perf_event_attr GetMapping(PerfCount::CountType type) __attribute__((unused));

perf_event_attr
GetMapping(PerfCount::CountType type) {
  perf_event_attr attr;
  switch (type) {
    case PerfCount::PERF_COUNT_TYPE_HW_CPU_CYCLES:
      attr.type = PERF_TYPE_HARDWARE;
      attr.config = PERF_COUNT_HW_CPU_CYCLES;
      break;
    case PerfCount::PERF_COUNT_TYPE_HW_INSTRUCTIONS:
      attr.type = PERF_TYPE_HARDWARE;
      attr.config = PERF_COUNT_HW_INSTRUCTIONS;
      break;
    case PerfCount::PERF_COUNT_TYPE_HW_CACHE_REFERENCES:
      attr.type = PERF_TYPE_HARDWARE;
      attr.config = PERF_COUNT_HW_CACHE_REFERENCES;
      break;
    case PerfCount::PERF_COUNT_TYPE_HW_CACHE_MISSES:
      attr.type = PERF_TYPE_HARDWARE;
      attr.config = PERF_COUNT_HW_CACHE_MISSES;
      break;
    case PerfCount::PERF_COUNT_TYPE_HW_BRANCH_INSTRUCTIONS:
      attr.type = PERF_TYPE_HARDWARE;
      attr.config = PERF_COUNT_HW_BRANCH_INSTRUCTIONS;
      break;
    case PerfCount::PERF_COUNT_TYPE_HW_BRANCH_MISSES:
      attr.type = PERF_TYPE_HARDWARE;
      attr.config = PERF_COUNT_HW_BRANCH_MISSES;
      break;
    case PerfCount::PERF_COUNT_TYPE_HW_BUS_CYCLES:
      attr.type = PERF_TYPE_HARDWARE;
      attr.config = PERF_COUNT_HW_BUS_CYCLES;
      break;
    case PerfCount::PERF_COUNT_TYPE_HW_CACHE_L1I_READ_ACCESS:
      attr.type = PERF_TYPE_HW_CACHE;
            attr.config = PERF_COUNT_HW_CACHE_L1I
                          | (PERF_COUNT_HW_CACHE_OP_READ << 8)
                          | (PERF_COUNT_HW_CACHE_RESULT_ACCESS << 16);
            break;
    case PerfCount::PERF_COUNT_TYPE_HW_CACHE_L1I_READ_MISS:
      attr.type = PERF_TYPE_HW_CACHE;
            attr.config = PERF_COUNT_HW_CACHE_L1I
                          | (PERF_COUNT_HW_CACHE_OP_READ << 8)
                          | (PERF_COUNT_HW_CACHE_RESULT_MISS << 16);
            break;
    case PerfCount::PERF_COUNT_TYPE_HW_CACHE_L1I_WRITE_ACCESS:
      attr.type = PERF_TYPE_HW_CACHE;
            attr.config = PERF_COUNT_HW_CACHE_L1I
                          | (PERF_COUNT_HW_CACHE_OP_WRITE << 8)
                          | (PERF_COUNT_HW_CACHE_RESULT_ACCESS << 16);
            break;
    case PerfCount::PERF_COUNT_TYPE_HW_CACHE_L1I_WRITE_MISS:
      attr.type = PERF_TYPE_HW_CACHE;
            attr.config = PERF_COUNT_HW_CACHE_L1I
                          | (PERF_COUNT_HW_CACHE_OP_WRITE << 8)
                          | (PERF_COUNT_HW_CACHE_RESULT_MISS << 16);
            break;
    case PerfCount::PERF_COUNT_TYPE_HW_CACHE_L1I_PREFETCH_ACCESS:     // not working?
      break;
    case PerfCount::PERF_COUNT_TYPE_HW_CACHE_L1I_PREFETCH_MISS:     // not working?
      break;
    case PerfCount::PERF_COUNT_TYPE_HW_CACHE_L1D_READ_ACCESS:
      attr.type = PERF_TYPE_HW_CACHE;
            attr.config = PERF_COUNT_HW_CACHE_L1D
                          | (PERF_COUNT_HW_CACHE_OP_READ << 8)
                          | (PERF_COUNT_HW_CACHE_RESULT_ACCESS << 16);
            break;
    case PerfCount::PERF_COUNT_TYPE_HW_CACHE_L1D_READ_MISS:
      attr.type = PERF_TYPE_HW_CACHE;
            attr.config = PERF_COUNT_HW_CACHE_L1I
                          | (PERF_COUNT_HW_CACHE_OP_READ << 8)
                          | (PERF_COUNT_HW_CACHE_RESULT_MISS << 16);

            break;
    case PerfCount::PERF_COUNT_TYPE_HW_CACHE_L1D_WRITE_ACCESS:
      attr.type = PERF_TYPE_HW_CACHE;
            attr.config = PERF_COUNT_HW_CACHE_L1D
                          | (PERF_COUNT_HW_CACHE_OP_WRITE << 8)
                          | (PERF_COUNT_HW_CACHE_RESULT_ACCESS << 16);
            break;
    case PerfCount::PERF_COUNT_TYPE_HW_CACHE_L1D_WRITE_MISS:
      attr.type = PERF_TYPE_HW_CACHE;
            attr.config = PERF_COUNT_HW_CACHE_L1D
                          | (PERF_COUNT_HW_CACHE_OP_WRITE << 8)
                          | (PERF_COUNT_HW_CACHE_RESULT_MISS << 16);
            break;
    case PerfCount::PERF_COUNT_TYPE_HW_CACHE_L1D_PREFETCH_ACCESS:     // not working?
      break;
    case PerfCount::PERF_COUNT_TYPE_HW_CACHE_L1D_PREFETCH_MISS:    // not working?
    case PerfCount::PERF_COUNT_TYPE_HW_CACHE_LL_READ_ACCESS:
    case PerfCount::PERF_COUNT_TYPE_HW_CACHE_LL_READ_MISS:
    case PerfCount::PERF_COUNT_TYPE_HW_CACHE_LL_WRITE_ACCESS:
    case PerfCount::PERF_COUNT_TYPE_HW_CACHE_LL_WRITE_MISS:
    case PerfCount::PERF_COUNT_TYPE_HW_CACHE_LL_PREFETCH_ACCESS:    // not working?
    case PerfCount::PERF_COUNT_TYPE_HW_CACHE_LL_PREFETCH_MISS:    // not working?
    case PerfCount::PERF_COUNT_TYPE_HW_CACHE_ITLB_READ_ACCESS:
    case PerfCount::PERF_COUNT_TYPE_HW_CACHE_ITLB_READ_MISS:
    case PerfCount::PERF_COUNT_TYPE_HW_CACHE_ITLB_WRITE_ACCESS:
    case PerfCount::PERF_COUNT_TYPE_HW_CACHE_ITLB_WRITE_MISS:
    case PerfCount::PERF_COUNT_TYPE_HW_CACHE_ITLB_PREFETCH_ACCESS:  // not working?
    case PerfCount::PERF_COUNT_TYPE_HW_CACHE_ITLB_PREFETCH_MISS:    // not working?
    case PerfCount::PERF_COUNT_TYPE_HW_CACHE_DTLB_READ_ACCESS:
    case PerfCount::PERF_COUNT_TYPE_HW_CACHE_DTLB_READ_MISS:
    case PerfCount::PERF_COUNT_TYPE_HW_CACHE_DTLB_WRITE_ACCESS:
    case PerfCount::PERF_COUNT_TYPE_HW_CACHE_DTLB_WRITE_MISS:
    case PerfCount::PERF_COUNT_TYPE_HW_CACHE_DTLB_PREFETCH_ACCESS:   // not working?
    case PerfCount::PERF_COUNT_TYPE_HW_CACHE_DTLB_PREFETCH_MISS:    // not working?
    case PerfCount::PERF_COUNT_TYPE_SW_CPU_CLOCK:
    case PerfCount::PERF_COUNT_TYPE_SW_TASK_CLOCK:
      break;
    case PerfCount::PERF_COUNT_TYPE_SW_PAGE_FAULTS:
      attr.type = PERF_TYPE_SOFTWARE;
      attr.config = PERF_COUNT_SW_PAGE_FAULTS;
      break;
    case PerfCount::PERF_COUNT_TYPE_SW_CONTEXT_SWITCHES:
      attr.type = PERF_TYPE_SOFTWARE;
      attr.config = PERF_COUNT_SW_CONTEXT_SWITCHES;
      break;
    case PerfCount::PERF_COUNT_TYPE_SW_CPU_MIGRATIONS:
      attr.type = PERF_TYPE_SOFTWARE;
      attr.config = PERF_COUNT_SW_CPU_MIGRATIONS;
      break;
    case PerfCount::PERF_COUNT_TYPE_SW_PAGE_FAULTS_MIN:
    case PerfCount::PERF_COUNT_TYPE_SW_PAGE_FAULTS_MAJ:
    case PerfCount::PERF_COUNT_TYPE_SW_ALIGNMENT_FAULTS:
    case PerfCount::PERF_COUNT_TYPE_SW_EMULATION_FAULTS:
    case PerfCount::PERF_COUNT_TYPE_MAX:
      break;
    case PerfCount::PERF_COUNT_TYPE_INVALID:
      break;
  }
  return attr;
}

} // empty namespace

#ifdef ORION_PERF_COUNT

PerfCount::PerfCount(const std::vector<PerfCount::CountType> &perf_count_types):
    ctx_(perf_count_types.size()) {

  for (size_t i = 0; i < perf_count_types.size(); ++i) {
    ctx_[i].type = perf_count_types[i];
    ctx_[i].event = GetMapping(perf_count_types[i]);
        ctx_[i].event.read_format
            = PERF_FORMAT_TOTAL_TIME_ENABLED | PERF_FORMAT_TOTAL_TIME_RUNNING;
        ctx_[i].event.inherit = 1;
        ctx_[i].counter = 0;
  }

  for (size_t i = 0; i < perf_count_types.size(); i++) {
    pid_t pid = 0;
    int cpu = -1;

    ctx_[i].fd = sys_perf_event_open(&(ctx_[i].event), pid, cpu, -1, 0);
    CHECK(ctx_[i].fd >= 0) << "open failed i = " << i
                           << " fd = " << ctx_[i].fd
                           << " errno = " << errno;
  }

}

void PerfCount::Accumulate(bool additive) {
  for (auto &ctx : ctx_) {
    uint64_t count[3];
    uint64_t accum_count[3] = {0, 0, 0};

    if (ctx.fd < 0) continue;

    count[0] = count[1] = count[2] = 0;
    ssize_t len = read(ctx.fd, count, sizeof(count));
    CHECK(len == sizeof(count)) << "perf_count: error while reading stats";

    accum_count[0] += count[0];
    accum_count[1] += count[1];
    accum_count[2] += count[2];

    if (accum_count[2] == 0){
      // no event occurred at all
    } else {
      if (accum_count[2] < accum_count[1]) {
        // need to scale
                accum_count[0] =
                    (uint64_t) (
                        (double) accum_count[0]
                        * (double) accum_count[1]
                        / (double) accum_count[2] + 0.5);
      }
    }

    if (additive) {
      ctx.counter += accum_count[0];
      // due to the scaling, we may observe a negative increment
      if ((int64_t) ctx.counter < 0)
        ctx.counter = 0;
    } else
      ctx.counter -= accum_count[0];
  }
}

void PerfCount::Start() {
  Accumulate(false);
}

void PerfCount::Stop() {
  Accumulate(true);
}

void PerfCount::Reset() {
  for (auto &ctx : ctx_) {
    ctx.counter = 0;
  }
}

uint64_t PerfCount::GetByType(CountType type) const {
  for (const auto &ctx : ctx_)
    if (ctx.type == type)
      return ctx.counter;
  return (uint64_t) -1;
}

uint64_t PerfCount::GetByIndex(size_t index) const {
  return ctx_[index].counter;
}

#else
PerfCount::PerfCount(const std::vector<CountType> &perf_count_types) { }
void PerfCount::Start() { }
void PerfCount::Stop() { }
void PerfCount::Reset() { }
uint64_t PerfCount::GetByType(CountType type) const { return 0; }
uint64_t PerfCount::GetByIndex(size_t index) const { return 0; }
#endif

}
}
