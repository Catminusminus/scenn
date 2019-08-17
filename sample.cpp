#include <iostream>
#include <scenn/load.hpp>
#include <scenn/scenn.hpp>

SCENN_CONSTEXPR auto mini_mnist_test() {
  using namespace scenn;
  auto mnist_data = load_mini_mnist_data<double>();
  auto evaluation =
      SequentialNetwork(CrossEntropy(), DenseLayer<784, 196, double>(),
                        ActivationLayer<196, double>(Sigmoid()),
                        DenseLayer<196, 3, double>(),
                        ActivationLayer<3, double>(Softmax()))
          .train<100>(std::move(sprout::get<0>(mnist_data)), 20, 0.1)
          .evaluate(std::move(sprout::get<1>(mnist_data)));
  return evaluation;
}

int main() {
  SCENN_CONSTEXPR auto evaluation = mini_mnist_test();
  std::cout << evaluation << std::endl;
}
