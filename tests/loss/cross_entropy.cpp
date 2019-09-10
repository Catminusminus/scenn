#include <scenn/loss/cross_entropy.hpp>
#include <scenn/matrix/matrix.hpp>

int main() {
  using namespace scenn;
  constexpr float arr[3] = {1.0f, 2.0f, 3.0f};
  constexpr auto vec1 = make_vector_from_array(arr);
  constexpr auto vec2 = vec1;
  [[maybe_unused]] constexpr auto a = CrossEntropy::loss_function(vec1, vec2);
  [[maybe_unused]] constexpr auto b = CrossEntropy::loss_derivative(vec1, vec2);
}
