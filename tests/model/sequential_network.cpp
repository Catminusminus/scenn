#include <scenn/model/sequential_network.hpp>
#include <scenn/layer.hpp>
#include <scenn/loss.hpp>
#include <scenn/activation.hpp>

int main() {
  using namespace scenn;
  [[maybe_unused]] SCENN_CONSTEXPR auto model = SequentialNetwork(
    CrossEntropy(), DenseLayer<2, 2, float>(), ActivationLayer<2, float>(Sigmoid())
  );
}
