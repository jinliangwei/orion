#include <orion/bosen/shape.hpp>
#include <iostream>
using namespace orion;
int main (int argc, char *argv[]) {
  size_t sdims[] = {4, 4, 2, 2};
  bosen::Shape<4> shape(sdims);
  size_t dims[4];
  shape.get_dims(14, dims);
  std::cout << dims[0] << " " << dims[1]
            << " " << dims[2] << " " << dims[3] << std::endl;

  return 0;
}
