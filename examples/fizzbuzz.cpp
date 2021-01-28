#include <iostream>
#include <scenn/load/mini_fizzbuzz.hpp>
#include <scenn/scenn.hpp>
#include <string>

template <std::size_t Size, class Model, class Data>
SCENN_CONSTEXPR auto predict(const Model& model, const Data& data) {
  std::array<int, Size> arr = {};
  auto i = 0;
  for (auto&& [x, y] : data.get_data()) {
    arr[i] = model.single_forward(x.transposed()).transposed().argmax();
    ++i;
  }
  return arr;
}

SCENN_CONSTEXPR auto mini_fizzbuzz_test() {
  using namespace scenn;
  auto [train_data, test_data, orig_data] = load_mini_fizzbuzz_data<double>();
  auto trained_model =
      SequentialNetwork(CrossEntropy(), DenseLayer<19, 32, double>(),
                        ActivationLayer<32, double>(ReLU()),
                        DenseLayer<32, 4, double>(),
                        ActivationLayer<4, double>(Softmax()))
          .train<100>(std::move(train_data), 300, 0.05);
  auto evaluation = trained_model.evaluate(test_data);
  return std::make_tuple(orig_data, predict<30>(trained_model, test_data),
                         evaluation);
}

int main() {
  SCENN_CONSTEXPR auto predictions = mini_fizzbuzz_test();

  const auto transform_fizzbuzz = [](auto prediction, auto value) {
    using namespace std::literals::string_literals;
    if (prediction == 0) {
      return "FizzBuzz"s;
    }
    if (prediction == 1) {
      return "Buzz"s;
    }
    if (prediction == 2) {
      return "Fizz"s;
    }
    return std::to_string(value);
  };
  auto [orig_data, predict_values, evaluation] = predictions;
  auto [x_train, x_test] = orig_data;
  for (auto i = 0; i < 30; ++i) {
    std::cout << x_test[i] << ": "
              << transform_fizzbuzz(predict_values[i], x_test[i]) << std::endl;
  }
  std::cout << "Acc: " << static_cast<float>(evaluation) / 30 << std::endl;
}
