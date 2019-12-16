#include <scenn/experimental/tensor/tensor3d.hpp>
#include <scenn/experimental/util/col2im.hpp>

int main() {
  using namespace scenn::experimental;
  constexpr float arr1[1][1][3] = {{{1, 4, 3}}};
  constexpr auto tensor1 = make_tensor3d_from_array(arr1);
  constexpr float arr2[2][2] = {{1, 2}, {2, 3}};
  constexpr auto mat2 = make_matrix_from_array(arr2);
  static_assert(col2im1d<2, 2, 1, 0, 1, 1, 3>(mat2) == tensor1);
}