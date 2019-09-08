#ifndef SCENN_EXPERIMENTALS_MODEL_SEQUENTIAL_NETWORK_HPP
#define SCENN_EXPERIMENTALS_MODEL_SEQUENTIAL_NETWORK_HPP

#include <scenn/util.hpp>
#include <sprout/tuple.hpp>

namespace scenn::experimentals {
template <class LossFunction, class... Layers>
class SequentialNetwork {
  LossFunction loss;
  using T = sprout::tuple<Layers...>;
  T layers;

 public:
  SCENN_CONSTEXPR SequentialNetwork(LossFunction &&loss, Layers &&... layers)
      : loss(loss), layers(sprout::make_tuple(layers...)){};
  SCENN_CONSTEXPR SequentialNetwork(const LossFunction &loss,
                                    const Layers &... layers)
      : loss(loss), layers(sprout::make_tuple(layers...)){};

 private:
  template <size_t index>
  SCENN_CONSTEXPR auto get_forward_input() {
    if constexpr (index == 0) {
      return sprout::get<index>(layers).input_data;
    } else {
      return sprout::get<index - 1>(layers).output_data;
    }
  }
  template <size_t index>
  SCENN_CONSTEXPR auto get_backward_input() {
    if constexpr (index + 1 == std::tuple_size_v<T>) {
      return sprout::get<index>(layers).input_delta;
    } else {
      return sprout::get<index + 1>(layers).output_delta;
    }
  }

  template <std::size_t index = 0, class I>
  SCENN_CONSTEXPR auto update_impl_impl(I rate) {
    if constexpr (index < std::tuple_size_v<T>) {
      sprout::get<index>(layers).update_params(rate);
      update_impl_impl<index + 1>(rate);
    }
  }

  template <class I>
  SCENN_CONSTEXPR auto update_params(I rate) {
    update_impl_impl(rate);
  }

  template <std::size_t index = 0>
  SCENN_CONSTEXPR auto clear_impl_impl() {
    if constexpr (index < std::tuple_size_v<T>) {
      sprout::get<index>(layers).clear_deltas();
      clear_impl_impl<index + 1>();
    }
  }
  SCENN_CONSTEXPR auto clear_deltas() { clear_impl_impl(); }
  template <class Train, class I>
  SCENN_CONSTEXPR auto update(Train &&mini_batch, I &&learning_rate) {
    auto mini_learning_rate =
        learning_rate / sprout::get<0>(mini_batch.shape());
    update_params(mini_learning_rate);
    clear_deltas();
  }

  template <std::size_t index = 0>
  SCENN_CONSTEXPR auto forward_impl() {
    if constexpr (index < std::tuple_size_v<T>) {
      sprout::get<index>(layers).forward(get_forward_input<index>());
      forward_impl<index + 1>();
    }
  }

  template <std::size_t index = std::tuple_size_v<T> - 1>
  SCENN_CONSTEXPR auto backward_impl() {
    sprout::get<index>(layers).backward(get_forward_input<index>(),
                                        get_backward_input<index>());
    if constexpr (index > 0) {
      backward_impl<index - 1>();
    }
  }

  template <class Train>
  SCENN_CONSTEXPR auto forward_backward(Train &&mini_batch) {
    for (auto &&[x, y] : mini_batch.get_data()) {
      sprout::get<0>(layers).input_data = x.transposed();
      forward_impl();
      sprout::get<std::tuple_size_v<T> - 1>(layers).input_delta =
          loss.loss_derivative(
              sprout::get<std::tuple_size_v<T> - 1>(layers).output_data,
              y.transposed());
      backward_impl();
#ifdef SCENN_DISABLE_CONSTEXPR
      std::cout << loss.loss_function(
                       single_forward(x.transposed()).transposed(), y)
                << std::endl;
#endif
    }
  }

  template <class Train, class I>
  SCENN_CONSTEXPR auto train_batch(Train &&mini_batch, I learning_rate) {
    forward_backward(mini_batch);
    update(std::forward<Train>(mini_batch), learning_rate);
  }

  template <std::size_t Index = 0, std::size_t Upper, std::size_t Interval,
            class Train, class T>
  SCENN_CONSTEXPR auto train_impl(Train &&training_data, T &&learning_rate) {
    if constexpr (Index < Upper) {
      auto data = training_data.template slice<Index, Index + Interval>();
      train_batch(std::move(data), learning_rate);
      train_impl<Index + Interval, Upper, Interval, Train, T>(
          std::forward<Train>(training_data), std::forward<T>(learning_rate));
    }
  }

 public:
  template <std::size_t MiniBatchSize, class Train, class T>
  SCENN_CONSTEXPR auto train(Train &&training_data, std::size_t epochs,
                             T &&learning_rate) {
    constexpr auto N = std::remove_reference_t<Train>::length();
    for (std::size_t epoch = 0; epoch < epochs; ++epoch) {
      auto shuffled_training_data = training_data.shuffle(epoch);
      train_impl<0, N, MiniBatchSize, std::remove_reference_t<Train> &&, T>(
          std::move(shuffled_training_data), std::forward<T>(learning_rate));
    }
    return *this;
  }
  template <class Test>
  SCENN_CONSTEXPR auto single_forward(Test &&x) {
    sprout::get<0>(layers).input_data = std::forward<Test>(x);
    forward_impl();
    return sprout::get<std::tuple_size_v<T> - 1>(layers).output_data;
  }

  template <class Test>
  SCENN_CONSTEXPR auto evaluate(Test &&test_data) {
    auto sum = 0;
    for (auto &&[x, y] : std::forward<Test>(test_data).get_data()) {
      if (single_forward(x.transposed()).transposed().argmax() == y.argmax())
        ++sum;
    }
    return sum;
  }
};
}  // namespace scenn::experimentals
#endif
