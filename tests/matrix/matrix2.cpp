#include <iostream>
#include <scenn/matrix/matrix.hpp>
int main() {
  using namespace scenn;
  // vector tests
  constexpr float arr1d[1] = {5.0};
  constexpr auto vec = make_vector_from_array(arr1d);
  static_assert(is_same_value(vec.to_value(), 5.0F));
  static_assert(is_same_value(make_vector_from_array(arr1d).to_value(), 5.0F));

  constexpr float arr1d1[3] = {1.0, 2.2, -3.4};
  constexpr float arr1d2[3] = {5.5, -100.2, 10.94};

  constexpr auto vec1 = make_vector_from_array(arr1d1);
  constexpr auto vec2 = make_vector_from_array(arr1d2);

  static_assert(is_same_value(vec1[1], 2.2F));
  static_assert(is_same_value(vec2(2), 10.94F));
  static_assert(vec1.argmax() == 1);
}