# Sequential ConstExpr Neural Network

Proof of concept of sequential models of neural networks in compile time.

<strong>Alpha Stage</strong>

## What is this?

You can build your own constexpr neural networks which run in compile time easily.

```cpp
// training and evaluating an model in compile time
constexpr auto mnist_data = scenn::load_mini_mnist_data();
constexpr auto evaluation = scenn::SequentialNetwork(
  CrossEntropy(),
  DenseLayer(784, 196),
  ActivationLayer(Sigmoid(), 196),
  DenseLayer(196, 3),
  ActivationLayer(Sigmoid(), 3)
).train(
  std::move(std::get<0>(mnist_data)), 10, 100, 0.1
).evaluate(
  std::move(std::get<1>(test_data))
);


std::cout << evaluation << std::end;
```

## How to use?

<strong>Now writing...</strong>

## Limitaion

So many limitaions exist...

- Only 1D data can be used.
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
