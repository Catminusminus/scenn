#include <scenn/activation/leaky_relu.hpp>
#include <scenn/matrix.hpp>

int main() {
  using namespace scenn;
  constexpr auto alpha = 0.3f;
  constexpr LeakyReLU l(alpha);
  constexpr float arr[3] = {1.0, 2.0, 3.0};
  constexpr auto mat1 = make_vector_from_array(arr);
  constexpr auto mat2 = mat1.fmap([](auto&& x) { return leaky_relu(x, alpha); });
  static_assert(l.activate(mat1) == mat2);

  constexpr auto mat3 =
      mat1.fmap([](auto&& x) { return leaky_relu_prime(x, alpha); }) * mat1;
  static_assert(l.calc_backward_pass(mat1, mat1) == mat3);
}
