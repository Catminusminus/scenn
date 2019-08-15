#include <scenn/activation.hpp>
#include <scenn/layer.hpp>
#include <scenn/loss.hpp>
#include <scenn/model/sequential_network.hpp>

int main() {
  using namespace scenn;
  [[maybe_unused]] SCENN_CONSTEXPR auto model =
      SequentialNetwork(BinaryCrossEntropy(), DenseLayer<2, 2, float>(),
                        ActivationLayer<2, float>(Sigmoid()));
  [[maybe_unused]] SCENN_CONSTEXPR auto model2 =
      SequentialNetwork(CrossEntropy(), DenseLayer<3, 3, double>(),
                        ActivationLayer<3, double>(Softmax()));
}
