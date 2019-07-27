#include <scenn/dataset/dataset.hpp>

int main() {
  using namespace scenn;
  constexpr float x_arr[3][2] = {{1.0, 2.0}, {-3.0, -4.0}, {-5.0, 6.0}};
  constexpr float y_arr[3][2] = {{1.0, 0.0}, {0.0, 1.0}, {0.0, 1.0}};
  constexpr Dataset dataset{make_matrix_from_array(std::move(x_arr)), make_matrix_from_array(std::move(y_arr))};

  constexpr float x_arr2[2][2] = {{-3.0, -4.0}, {-5.0, 6.0}};
  constexpr float y_arr2[2][2] = {{0.0, 1.0}, {0.0, 1.0}};
  constexpr Dataset dataset2{make_matrix_from_array(std::move(x_arr2)), make_matrix_from_array(std::move(y_arr2))};

  static_assert(dataset.slice<1, 2>().get_data()[0] == dataset2.get_data()[0]);
}
