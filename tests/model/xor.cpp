#include <scenn/model/sequential_network.hpp>
#include <scenn/layer.hpp>
#include <scenn/loss.hpp>
#include <scenn/activation.hpp>
#include <scenn/dataset.hpp>

constexpr auto test() {
  using namespace scenn;
  float X_arr[4][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
  float Y_arr[4][2] = {{1, 0}, {0, 1}, {0, 1}, {1, 0}};
  auto X = make_matrix_from_array(std::move(X_arr));
  auto Y = make_matrix_from_array(std::move(Y_arr));
  auto trained_model = SequentialNetwork(
    CrossEntropy(), DenseLayer<2, 2, float>(), ActivationLayer<2, float>(Sigmoid()), DenseLayer<2, 2, float>(), ActivationLayer<2, float>(Sigmoid())
  ).train<2>(Dataset(make_matrix_from_array(std::move(X_arr)), make_matrix_from_array(std::move(Y_arr))), 10, 0.1);
  return trained_model;
}
int main() {
  constexpr auto model = test();
}
