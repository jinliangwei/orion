#include <orion/vector_occ.hpp>
#include <orion/version.hpp>

#include <iostream>

orion::VectorOCC<int>::Key keys[10];
orion::VectorOCC<int>::Update updates[10];
int *values[10];

int main(int argc __attribute__((unused)), char *argv[] __attribute__((unused))) {
  orion::VectorOCC<int> v(100);

  keys[0].group = 0;
  keys[0].group_offset = 0;
  keys[0].key = 0;

  keys[1].group = 0;
  keys[1].group_offset = 1;
  keys[1].key = 1;

  keys[2].group = 0;
  keys[2].group_offset = 2;
  keys[2].key = 2;

  values[0] = new int[16];
  v.Get(0, keys, 3, values);

  for (int i = 0; i < 1; i++) {
    for (int j = 0; j < 16; j++) {
      std::cout << values[i][j] << " ";
    }
    std::cout << std::endl;
  }

  updates[0].key = 0;
  updates[0].delta = 10;

  updates[1].key = 1;
  updates[1].delta = 2;

  updates[2].key = 2;
  updates[2].delta = 1;


  v.Inc(0, updates, 3);
  v.Get(0, keys, 3, values);

  for (int i = 0; i < 1; i++) {
    for (int j = 0; j < 16; j++) {
      std::cout << values[i][j] << " ";
    }
    std::cout << std::endl;
  }

  delete[] values[0];
  return 0;
}
