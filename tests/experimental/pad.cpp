#include <scenn/experimental/tensor/tensor3d.hpp>
#include <scenn/experimental/util/pad.hpp>

int main() {
  using namespace scenn::experimental;
  constexpr float arr1[3][2][1] = {{{1}, {2}}, {{3}, {4}}, {{5}, {6}}};
  constexpr auto tensor1 = make_tensor3d_from_array(arr1);
  constexpr float arr2[3][4][1] = {
      {{0}, {1}, {2}, {0}}, {{0}, {3}, {4}, {0}}, {{0}, {5}, {6}, {0}}};
  constexpr auto tensor2 = make_tensor3d_from_array(arr2);
  static_assert(pad<1>(tensor1) == tensor2);
}