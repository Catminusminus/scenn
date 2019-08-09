#include <scenn/layer/dense_layer.hpp>

int main() {
  using namespace scenn;
  [[maybe_unused]] constexpr auto dense_layer = DenseLayer<3, 2, float>();
}