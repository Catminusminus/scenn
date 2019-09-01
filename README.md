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
          .train<2>(dataset, 2000, 0.1);
  return trained_model.evaluate(std::move(dataset));
}

int main() {
  SCENN_CONSTEXPR auto evaluation = test();
  std::cout << evaluation << std::endl; // 4
}

```
like the Python code with [Keras](https://github.com/keras-team/keras)
```python
X_train = numpy.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y_train = numpy.array([[1, 0], [0, 1], [0, 1], [1, 0]])
model = Sequential([
  Dense(4, input_dim=2),
  Activation('sigmoid'),
  Dense(2, input_dim=4),
  Activation('sigmoid')
])
model.compile(loss='binary_crossentropy', optimizer=SGD(0.1))
model.fit(X_train, Y_train, batch_size=2, epochs=2000)
# Omit the evaluation
```

<strong>MNIST EXAMPLE:</strong>
To avoid hitting the constexpr evaluation step limit, you need to patch `clang-patch.diff` to clang. In the below code, we use a subset of [MNIST dataset](http://yann.lecun.com/exdb/mnist/), with 100 train size, 10 test size and 3 classes. To generate the sub-dataset, use `tools/generate_mini_mnist.py`.
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
  std::cout << evaluation << std::endl; // we see 8
}
```

## How to use?

### Requirements

- [Sprout C++ Libraries](https://github.com/bolero-MURAKAMI/Sprout)

### Install

This library is a header-only library.

Include ```<scenn/scenn.hpp>``` and you can use all things provided by scenn except for ```load_mini_mnist_data```. To use `load_mini_mnist_data` function, you need to prepare the mnist sub-dataset by `tools/generate_mini_mnist.py` and include `<scenn/load.hpp>`.

### Quick Start
```
git clone https://github.com/bolero-MURAKAMI/Sprout.git
git clone https://github.com/Catminusminus/scenn.git

export SPROUT_PATH=./Sprout/
export SCENN_PATH=./scenn/

// Run the xor example
clang++ ./scenn/tests/model/xor.cpp -Wall -Wextra -I$SPROUT_PATH -I$SCENN_PATH -std=gnu++2a -fconstexpr-steps=-1

// After a few hours

./a.out // You will see 4.
```

### Features

For more details, see [documentation](
https://catminusminus.github.io/scenn-doc/) .
#### Model
- `scenn::SequentialNetwork`
    - `SequentialNetwork(loss_function, layers...)`
    - `train<MiniBatchSize>(training_data, epochs, learning_rate)`
    - `single_forwatd(test_data)`
    - `evaluate(test_data)`

#### Layers
- `scenn::DenseLayer<input_dim, output_dim, num_type>(seed)`
- `scenn::ActivationLayer<dim, num_type>(activation_function)`

#### Activation
- `scenn::Sigmoid()`
- `scenn::Softmax()`

#### Loss Function
- `scenn::MSE()`
- `scenn::BinaryCrossEntropy()`
- `scenn::CrossEntropy()`

## Limitaion

So many limitaions exist...

- Only 1D data can be used.
- Hittig the constexpr evaluation step limit easily.
  - Use the patch `clang-patch.diff` for newer clang based on https://github.com/ushitora-anqou/constexpr-nn/blob/master/clang.diff.
- You can run training and evaluating only in compile time, because runtime training and evaluating hit the stack size limit. We are planning to use the heap memory allocation in runtime (we will use std::vector in both cases in C++20).

## How it works?

Thanks to [Sprout C++ Libraries](https://github.com/bolero-MURAKAMI/Sprout)

## Important Disclosure

This project is inspired by a neural network in [dlgo](https://github.com/maxpumperla/deep_learning_and_the_game_of_go/tree/master/code/dlgo/nn).

scenn/matrix/matrix.hpp is based on https://github.com/ushitora-anqou/constexpr-nn/blob/master/main.cpp.
The license file is LICENSE.constexpr-nn.

## Author

Catminusminus

## LICENSE

MIT
