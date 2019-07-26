#include <scenn/matrix/matrix.hpp>
#include <iostream>
int main() {
  using namespace scenn;
  // non-vector tests
  constexpr float arr1[3][2] = {{1.0, -2.2}, {3.99, 4.3}, {-10.22, -3.0}};
  constexpr auto mat1 = make_matrix_from_array(arr1);

  constexpr float arr2[3][2] = {{4.0, 7.2}, {1.01, 0.7}, {15.22, 8.0}};
  constexpr auto mat2 = make_matrix_from_array(arr2);

  constexpr float arr3[3][2] = {{5, 5}, {5, 5}, {5, 5}};
  static_assert((mat1 + mat2) == make_matrix_from_array(arr3));

  constexpr float arr4[3][2] = {{5, 5}, {5, 5}, {5, 7}};
  static_assert((mat1 + mat2) != make_matrix_from_array(arr4));

  constexpr float arr5[3][2] = {{-3.0, -9.4}, {2.98, 3.6}, {-25.44, -11.0}};
  static_assert(is_same_value((mat1 - mat2)(1,1), make_matrix_from_array(arr5)(1,1)));

  constexpr float arr6[3][2] = {{-1.0, 2.2}, {-3.99, -4.3}, {10.22, 3.0}};
  static_assert(mat1 * -1 == make_matrix_from_array(arr6));

  constexpr float arr7[3][2] = {{2.0, 3.6}, {0.505, 0.35}, {7.61, 4.0}};
  static_assert(mat2 / 2.0 == make_matrix_from_array(arr7));

  static_assert(mat2.fmap([](auto&& x){ return x / 2.0; }) == make_matrix_from_array(arr7));

  constexpr float arr8[2][3] = {{4.0, 1.01, 15.22}, {7.2, 0.7, 8.0}};
  static_assert(mat2.transposed() == make_matrix_from_array(arr8));
}
