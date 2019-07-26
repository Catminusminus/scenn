#include <scenn/matrix/matrix.hpp>

int main() {
  // make_matrix_from_array
  constexpr int arr2d[2][2] = {{0, 0}, {0, 0}};
  [[maybe_unused]] constexpr auto mat = make_matrix_from_array(arr2d);
  constexpr int arr1d[2] = {1, 2};
  [[maybe_unused]] constexpr auto vec = make_vector_from_array(arr1d);
}
