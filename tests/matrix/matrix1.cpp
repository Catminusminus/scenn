#include <scenn/matrix/matrix.hpp>

int main() {
  using namespace scenn;
  // make_matrix_from_array works well
  constexpr int arr2d[2][2] = {{0, 0}, {0, 0}};
  [[maybe_unused]] constexpr auto mat = make_matrix_from_array(arr2d);

  // make_vector_from_array works well
  constexpr int arr1d[2] = {1, 2};
  [[maybe_unused]] constexpr auto vec = make_vector_from_array(arr1d);

  // make_zeros_from_pair works well
  [[maybe_unused]] constexpr auto zeros =
      make_zeros_from_pair<decltype(mat)::m, decltype(mat)::n,
                           decltype(mat)::value_type>();
}
