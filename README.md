# Sequential ConstExpr Neural Network

Proof of concept of sequential models of neural networks in compile time.

<strong>Alpha Stage</strong>

<strong>Any contributions are greatly welcomed. Thank you.</strong>

## What is this?

You can build your own constexpr neural networks which run in compile time easily. For detailed information, see the [documentation](
https://catminusminus.github.io/scenn-doc/) (all information in this readme is also in the documentaion).

<strong>XOR Example:</strong>
```cpp
// training and evaluating an model in compile time
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
          .train<2>(dataset,
                    2000, 0.1);
  return trained_model.evaluate(std::move(dataset));
}

int main() {
  SCENN_CONSTEXPR auto evaluation = test();
  std::cout << evaluation; // 4
}

```

<strong>WIP</strong>
The blow code should be worked, but fails due to hitting the constexpr evaluation step limit.
```cpp
// training and evaluating an model in compile time
SCENN_CONSTEXPR auto mini_mnist_test() {
  using namespace scenn;
  auto [train_data, test_data] = load_mini_mnist_data<double>();
  auto evaluation =
      SequentialNetwork(CrossEntropy(), DenseLayer<784, 196, double>(),
                        ActivationLayer<196, double>(Sigmoid()),
                        DenseLayer<196, 3, double>(),
                        ActivationLayer<3, double>(Softmax()))
          .train<100>(std::move(train_data), 20, 0.1)
          .evaluate(std::move(test_data));
  return evaluation;
}

int main() {
  SCENN_CONSTEXPR auto evaluation = mini_mnist_test();
  std::cout << evaluation << std::endl;
}
```

## How to use?

<strong>Now writing...</strong>
### Requirements

- [Sprout C++ Library](https://github.com/bolero-MURAKAMI/Sprout)

### Install

This library is a header-only library.

Include ```<scenn/scenn.hpp>``` and you can use all things provided by scenn except for ```load_mini_mnist_data```.

### Quick Start
```
git clone https://github.com/bolero-MURAKAMI/Sprout.git
git clone https://github.com/Catminusminus/scenn.git

export SPROUT_PATH=./Sprout/
export SCENN_PATH=./scenn/

clang++ ./scenn/tests/model/xor.cpp -Wall -Wextra -I$SPROUT_PATH -I$SCENN_PATH -std=gnu++2a -fconstexpr-steps=-1

// After a few hours

./a.out // You will see 4.
```

## Limitaion

So many limitaions exist...

- Only 1D data can be used.
- Hittig the constexpr evaluation step limit easily.
  - We are preparing a patch for newer clang based on https://github.com/ushitora-anqou/constexpr-nn/blob/master/clang.diff. clang-patch.diff in this repository works, but we think it includes some unintended changes.
- Now writing...

## How it works?

Thanks to [Sprout C++ Library](https://github.com/bolero-MURAKAMI/Sprout)

## Important Disclosure

This project is inspired by a neural network in [dlgo](https://github.com/maxpumperla/deep_learning_and_the_game_of_go/tree/master/code/dlgo/nn).

scenn/matrix/matrix.hpp is based on https://github.com/ushitora-anqou/constexpr-nn/blob/master/main.cpp.
The license file is LICENSE.constexpr-nn.

## Author

Catminusminus

## LICENSE

MIT
