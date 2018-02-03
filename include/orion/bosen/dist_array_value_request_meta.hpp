#pragma once
namespace orion {
namespace bosen {
struct DistArrayValueRequestMeta {
  int32_t requester_id {-1};
  bool is_requester_executor {true};
};

}
}
