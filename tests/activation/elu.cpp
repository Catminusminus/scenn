#include <scenn/activation/elu.hpp>
#include <scenn/matrix.hpp>

int main() {
  using namespace scenn;
  constexpr auto alpha = 1.0f;
  constexpr ELU e(alpha);
  constexpr float arr[3] = {1.0, 2.0, 3.0};
  constexpr auto mat1 = make_vector_from_array(arr);
  constexpr auto mat2 = mat1.fmap([](auto&& x) { return elu(x, alpha); });
  static_assert(e.activate(mat1) == mat2);

  constexpr auto mat3 =
      mat1.fmap([](auto&& x) { return elu_prime(x, alpha); }) * mat1;
  static_assert(e.calc_backward_pass(mat1, mat1) == mat3);
}
