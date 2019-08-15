#include <iostream>
#include <scenn/activation.hpp>
#include <scenn/dataset.hpp>
#include <scenn/layer.hpp>
#include <scenn/loss.hpp>
#include <scenn/model/sequential_network.hpp>

SCENN_CONSTEXPR auto test() {
  using namespace scenn;
  double X_arr[4][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
  double Y_arr[4][2] = {{1, 0}, {0, 1}, {0, 1}, {1, 0}};
  auto dataset = Dataset(make_matrix_from_array(std::move(X_arr)),
                         make_matrix_from_array(std::move(Y_arr)));
  auto trained_model =
      SequentialNetwork(BinaryCrossEntropy(), DenseLayer<2, 4, double>(),
                        ActivationLayer<4, double>(Sigmoid()),
                        DenseLayer<4, 2, double>(10),
                        ActivationLayer<2, double>(Sigmoid()))
          .train<2>(dataset, 2000, 0.1);
  return trained_model.evaluate(std::move(dataset));
}

int main() {
  SCENN_CONSTEXPR auto evaluation = test();
  std::cout << evaluation << std::endl;
}
