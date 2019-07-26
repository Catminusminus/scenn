# Sequential ConstExpr Neural Network

Proof of concept of sequential models of neural networks in compile time.

## What is this?

You can build your own constexpr neural networks which run in compile time easily.

```cpp
// training and evaluating in compile time
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

Enjoy!

## Limitaion

So much limitaion exist...

## How it works?

In fact, this is an of dlgo.
Thanks to Sprout C++ Library,

## Author

Catminusminus

## LiCENSE

Unlicense, except for scenn/matrix/matrix.hpp

scenn/matrix/matrix.hpp is under MIT License. See scenn/matrix/matrix.hpp and https://github.com/ushitora-anqou/constexpr-nn
