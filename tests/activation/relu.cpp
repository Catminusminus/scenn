#include <scenn/activation/relu.hpp>
#include <scenn/matrix.hpp>

int main() {
  using namespace scenn;
  constexpr ReLU r;
  constexpr float arr[3] = {1.0, 2.0, 3.0};
  constexpr auto mat1 = make_vector_from_array(arr);
  constexpr auto mat2 = mat1.fmap([](auto&& x) { return relu(x); });
  static_assert(r.activate(mat1) == mat2);

  constexpr auto mat3 =
      mat1.fmap([](auto&& x) { return relu_prime(x); }) * mat1;
  static_assert(r.calc_backward_pass(mat1, mat1) == mat3);
}
