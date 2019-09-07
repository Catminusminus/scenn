#include <scenn/activation/selu.hpp>
#include <scenn/matrix.hpp>

int main() {
  using namespace scenn;
  constexpr SeLU<float> s;
  constexpr float alpha = 1.67326324;
  constexpr float scale = 1.05070098;
  constexpr float arr[3] = {1.0, 2.0, 3.0};
  constexpr auto mat1 = make_vector_from_array(arr);
  constexpr auto mat2 = mat1.fmap([](auto&& x) { return selu(x, alpha, scale); });
  static_assert(s.activate(mat1) == mat2);

  constexpr auto mat3 =
      mat1.fmap([](auto&& x) { return selu_prime(x, alpha, scale); }) * mat1;
  static_assert(s.calc_backward_pass(mat1, mat1) == mat3);
}
