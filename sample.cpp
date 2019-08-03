#include <iostream>
#include <load.hpp>
#include <scenn.hpp>

int main() {
  constexpr auto mnist_data = scenn::load_mini_mnist_data();
  constexpr auto evaluation =
      scenn::SequentialNetwork(
          CrossEntropy(), DenseLayer(784, 196), ActivationLayer(Sigmoid(), 196),
          DenseLayer(196, 3), ActivationLayer(Sigmoid(), 3))
          .train<100>(std::move(std::get<0>(mnist_data)), 10, 0.1)
          .evaluate(std::move(std::get<1>(mnist_data)));
  std::cout << evaluation << std::end;
}
