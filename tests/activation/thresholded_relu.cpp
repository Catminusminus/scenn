#include <scenn/activation/thresholded_relu.hpp>
#include <scenn/matrix.hpp>

int main() {
  using namespace scenn;
  constexpr auto theta = 1.0f;
  constexpr ThresholdedReLU t(theta);
  constexpr float arr[3] = {1.0, 2.0, 3.0};
  constexpr auto mat1 = make_vector_from_array(arr);
  constexpr auto mat2 = mat1.fmap([](auto&& x) { return thresholded_relu(x, theta); });
  static_assert(t.activate(mat1) == mat2);

  constexpr auto mat3 =
      mat1.fmap([](auto&& x) { return thresholded_relu_prime(x, theta); }) * mat1;
  static_assert(t.calc_backward_pass(mat1, mat1) == mat3);
}
