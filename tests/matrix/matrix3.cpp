#include <scenn/matrix/matrix.hpp>
#include <iostream>
int main() {
  using namespace scenn;
  // non-vector tests
  constexpr float arr2d1[3][2] = {{1.0, -2.2}, {3.99, 4.3}, {-10.22, -3.0}};
  constexpr auto mat1 = make_matrix_from_array(arr2d1);

  constexpr float arr2d2[3][2] = {{4.0, 7.2}, {1.01, 0.7}, {15.22, 8.0}};
  constexpr auto mat2 = make_matrix_from_array(arr2d2);

  constexpr float arr2d3[3][2] = {{5, 5}, {5, 5}, {5, 5}};
  static_assert((mat1 + mat2) == make_matrix_from_array(arr2d3));

  constexpr float arr2d4[3][2] = {{5, 5}, {5, 5}, {5, 7}};
  static_assert((mat1 + mat2) != make_matrix_from_array(arr2d4));

  constexpr float arr2d5[3][2] = {{-3.0, -9.4}, {2.98, 3.6}, {-25.44, -11.0}};
  static_assert(is_same_value((mat1 - mat2)(1,1), make_matrix_from_array(arr2d5)(1,1)));

  constexpr float arr2d6[3][2] = {{-1.0, 2.2}, {-3.99, -4.3}, {10.22, 3.0}};
  static_assert(mat1 * -1 == make_matrix_from_array(arr2d6));

  constexpr float arr2d7[3][2] = {{2.0, 3.6}, {0.505, 0.35}, {7.61, 4.0}};
  static_assert(mat2 / 2.0 == make_matrix_from_array(arr2d7));

  static_assert(mat2.fmap([](auto&& x){ return x / 2.0; }) == make_matrix_from_array(arr2d7));
}