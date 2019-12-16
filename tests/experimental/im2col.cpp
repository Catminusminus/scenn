#include <scenn/experimental/tensor/tensor3d.hpp>
#include <scenn/experimental/util/im2col.hpp>

int main() {
  using namespace scenn::experimental;
  constexpr float arr1[1][1][3] = {{{1, 2, 3}}};
  constexpr auto tensor1 = make_tensor3d_from_array(arr1);
  constexpr auto mat1 = im2col1d<2, 2, 1, 0>(tensor1);
  constexpr float arr2[2][2] = {{1, 2}, {2, 3}};
  constexpr auto mat2 = make_matrix_from_array(arr2);
  static_assert(mat1 == mat2);
}
