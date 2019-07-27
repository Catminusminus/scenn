#ifndef SCENN_MODEL_SEQUENTIAL_NETWORK_HPP
#define SCENN_MODEL_SEQUENTIAL_NETWORK_HPP

namespace scenn {
template <class T, class LossFunction, class... Layers>
struct SequentialNetwork {
  T layers;
  LossFunction loss;
  constexpr SequentialNetwork(LossFunction &&loss, Layers &&... layers)
      : loss(std::forward<LossFunction>(loss)),
        layers(std::make_tuple(std::forward<Layers>(layers)...)){};
  template <size_t index>
  constexpr auto get_forward_input() {
    if (index == 0) return std::get<0>(layers).input_data;
    return std::get<index - 1>(layers).output_data;
  }
  template <size_t index>
  constexpr auto get_backward_input() {
    if (index + 1 == std::tuple_size_v<layers>)
      return std::get<index>(layers).input_delta;
    return std::get<index + 1>(layers).output_delta;
  }

  template <class I, std::size_t index>
  constexpr auto update_impl_impl(T &new_layers, T &old_layers, I rate) {
    std::get<index>(new_layers) =
        std::get<index>(old_layers).update_params(rate);
  }

  template <class I, std::size_t... Indices>
  constexpr auto update_impl(T &new_layers, T &old_layers, I rate,
                             std::index_sequence<Indices...>) {
    update_impl_impl<I, Indices>(new_layers, old_layer, rate)...;
  }

  template <class I>
  constexpr auto update_params(T &new_layers, T &old_layers, I rate) {
    update_impl(new_layers, old_layers, rate,
                std::make_index_sequence<std::tuple_size_v<T>>{});
  }

  template <std::size_t index>
  constexpr auto clear_impl_impl(T &new_layers, T &old_layers) {
    std::get<index>(new_layers) = std::get<index>(old_layers).clear_deltas();
  }

  template <class I, std::size_t... Indices>
  constexpr auto clear_impl(T &new_layers, T &old_layers, I rate,
                            std::index_sequence<Indices...>) {
    clear_impl_impl<Indices>(new_layers, old_layers, Indices)...;
  }

  constexpr auto clear_deltas(T &new_layers, T &old_layers) {
    clear_impl(new_layers, old_layers,
               std::make_index_sequence<std::tuple_size_v<T>>{});
  }

  template <class Train, class I>
  constexpr auto update(Train &&mini_batch, I &&learning_rate) {
    auto mini_learning_rate = learning_rate / std::get<0>(mini_batch.shape());
    T new_layers;
    T ret_layers;
    update_params(new_layers, old_layers, mini_learning_rate);
    clear_deltas(ret_layers, new_layers);
    return std::make_from_tuple<SequentialNetwork>(
        std::tuple_cat(std::tie(loss), ret_layers));
  }

  template <std::size_t index = 0>
  constexpr auto forward_impl() {
    if constexpr (index < std::tuple_size_v<T>) {
      std::get<index>(layers) = layers.forward(get_forward_input<index>());
      forward_impl<index + 1>();
    }
  }

  template <std::size_t index = std::tuple_size_v<T> - 1>
  constexpr auto backward_impl() {
    std::get<index>(layers) = layers.backward(get_forward_input<index>(),
                                              get_backward_input<index>());
    if constexpr (index > 0) {
      backward_impl<index - 1>();
    }
  }

  template <class Train>
  constexpr auto forward_backward(Train &&mini_batch) {
    for (auto &&[x, y] : mini_batch.get_date()) {
      std::get<0>(layers) = std::get<0>(layers).make_by_input_data(x);
      forward_impl();
      std::get<std::tuple_size_v<T> - 1>(layers) =
          std::get<std::tuple_size_v<T> - 1>(layers).make_by_input_delta(
              loss.loss_derivative(
                  std::get<std::tuple_size_v<T> - 1>(layers).output_data, y));
      backward_impl();
    }
  }

  template <class Train, class I>
  constexpr auto train_batch(Train &&mini_batch, I learning_rate) {
    forward_backward(mini_batch);
    return update(mini_batch, learning_rate);
  }

  template <class Train, class T>
  constexpr auto train(Train &&training_data, size_t epochs,
                       size_t mini_batch_size, T &&learning_rate) {
    auto n = training_data.length();
    for (auto epoch = 0; epoch < epochs; ++epoch) {
      shuffled_training_data = trainging_data.shuffle();
      auto trained = (*this);
      for (auto k = 0; k < n; k += mini_batch_size) {
        auto trained = trained.train_batch(
            shuffled_training_data.slice(k, k + mini_batch_size),
            learning_rate);
      }
    }
    return trained;
  }
  template <class Test>
  constexpr auto single_forward(Test &&x) {
    std::get<0>(layers) = std::get<0>(layers).make_by_input_data(x);
    forward_impl();
    return std::get<std::tuple_size_v<T> - 1>(layers).output_data;
  }

  template <class Test>
  constexpr auto evaluate(Test &&test_data) {
    auto sum = 0;
    for (const auto &&[x, y] : test_data) {
      if (single_forward(x).argmax() == y.argmax())
        ;
    }
    return sum;
  }
};
}  // namespace scenn
#endif
