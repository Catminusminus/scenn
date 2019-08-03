#ifndef SCENN_MODEL_SEQUENTIAL_NETWORK_HPP
#define SCENN_MODEL_SEQUENTIAL_NETWORK_HPP

#include <sprout/tuple.hpp>

namespace scenn {
template <class LossFunction, class... Layers>
struct SequentialNetwork {
  LossFunction loss;
  using T = sprout::tuple<Layers...>;
  T layers;
  constexpr SequentialNetwork(LossFunction &&loss, Layers &&... layers)
      : loss(loss), layers(sprout::make_tuple(layers...)){};
  constexpr SequentialNetwork(const LossFunction &loss,
                              const Layers &... layers)
      : loss(loss), layers(sprout::make_tuple(layers...)){};
  template <size_t index>
  constexpr auto get_forward_input() const {
    if constexpr (index == 0) {
      return sprout::get<index>(layers).input_data;
    } else {
      return sprout::get<index - 1>(layers).output_data;
    }
  }
  template <size_t index>
  constexpr auto get_backward_input() const {
    if constexpr (index + 1 == std::tuple_size_v<T>) {
      return sprout::get<index>(layers).input_delta;
    } else {
      return sprout::get<index + 1>(layers).output_delta;
    }
  }

  template <std::size_t index = 0, class I>
  constexpr auto update_impl_impl(T &new_layers, I rate) const {
    if constexpr (index < std::tuple_size_v<T>) {
      sprout::get<index>(new_layers) =
          sprout::get<index>(layers).update_params(rate);
      update_impl_impl<index + 1>(new_layers, rate);
    }
  }
  /**
  template <class I, std::size_t... Indices>
  constexpr auto update_impl(T &new_layers, T &old_layers, I rate,
                             std::index_sequence<Indices...>) const {
    update_impl_impl<I, Indices>(new_layers, old_layer, rate)...;
  }
  */
  template <class I>
  constexpr auto update_params(T &new_layers, I rate) const {
    /**
    update_impl(new_layers, old_layers, rate,
                std::make_index_sequence<std::tuple_size_v<T>>{});
    */
    update_impl_impl(new_layers, rate);
  }

  template <std::size_t index = 0>
  constexpr auto clear_impl_impl(T &new_layers, T &old_layers) const {
    if constexpr (index < std::tuple_size_v<T>) {
      sprout::get<index>(new_layers) =
          sprout::get<index>(old_layers).clear_deltas();
      clear_impl_impl<index + 1>(new_layers, old_layers);
    }
  }
  /**
  template <class I, std::size_t... Indices>
  constexpr auto clear_impl(T &new_layers, T &old_layers, I rate,
                            std::index_sequence<Indices...>) const {
    clear_impl_impl<Indices>(new_layers, old_layers, Indices)...;
  }

  constexpr auto clear_deltas(T &new_layers, T &old_layers) const {
    clear_impl(new_layers, old_layers,
               std::make_index_sequence<std::tuple_size_v<T>>{});
  }
  */
  constexpr auto clear_deltas(T &new_layers, T &old_layers) const {
    clear_impl_impl(new_layers, old_layers);
  }
  template <class Train, class I>
  constexpr auto update(Train &&mini_batch, I &&learning_rate) const {
    auto mini_learning_rate =
        learning_rate / sprout::get<0>(mini_batch.shape());
    auto new_layers = layers;
    auto ret_layers = layers;
    update_params(new_layers, mini_learning_rate);
    clear_deltas(ret_layers, new_layers);
    return sprout::make_from_tuple<SequentialNetwork>(
        sprout::tuple_cat(sprout::tie(loss), ret_layers));
  }

  template <std::size_t index = 0>
  constexpr auto forward_impl(T &new_layers, T &old_layers) const {
    if constexpr (index < std::tuple_size_v<T>) {
      sprout::get<index>(new_layers) =
          sprout::get<index>(old_layers).forward(get_forward_input<index>());
      forward_impl<index + 1>(new_layers, old_layers);
    }
  }

  template <std::size_t index = std::tuple_size_v<T> - 1>
  constexpr auto backward_impl(T &new_layers, T &old_layers) const {
    sprout::get<index>(new_layers) =
        sprout::get<index>(old_layers)
            .backward(get_forward_input<index>(), get_backward_input<index>());
    if constexpr (index > 0) {
      backward_impl<index - 1>(new_layers, old_layers);
    }
  }

  template <class Train>
  constexpr auto forward_backward(Train &&mini_batch) const {
    auto new_layers = layers;
    for (auto &&[x, y] : mini_batch.get_data()) {
      sprout::get<0>(new_layers) =
          sprout::get<0>(new_layers).make_by_input_data(x.transposed());
      forward_impl(new_layers, new_layers);
      sprout::get<std::tuple_size_v<T> - 1>(new_layers) =
          sprout::get<std::tuple_size_v<T> - 1>(new_layers)
              .make_by_input_delta(loss.loss_derivative(
                  sprout::get<std::tuple_size_v<T> - 1>(new_layers).output_data,
                  y.transposed()));
      backward_impl(new_layers, new_layers);
    }
    return sprout::make_from_tuple<SequentialNetwork>(
        sprout::tuple_cat(sprout::tie(loss), new_layers));
  }

  template <class Train, class I>
  constexpr auto train_batch(Train &&mini_batch, I learning_rate) {
    return forward_backward(mini_batch)
        .update(std::forward<Train>(mini_batch), learning_rate);
  }

  template <std::size_t Index = 0, std::size_t Upper, std::size_t Interval,
            class Train, class T, class Model>
  constexpr auto train_impl(Train &&training_data, T &&learning_rate,
                            Model &model) {
    if constexpr (Index < Upper) {
      auto data = training_data.template slice<Index, Index + Interval>();
      model = model.train_batch(std::move(data), learning_rate);
      train_impl<Index + Interval, Upper, Interval, Train, T, Model>(
          std::forward<Train>(training_data), std::forward<T>(learning_rate),
          model);
    }
  }

  template <std::size_t MiniBatchSize, class Train, class T>
  constexpr auto train(Train &&training_data, std::size_t epochs,
                       T &&learning_rate) {
    constexpr auto N = Train::length();
    auto trained = (*this);
    for (std::size_t epoch = 0; epoch < epochs; ++epoch) {
      auto shuffled_training_data = training_data.shuffle();
      train_impl<0, N, MiniBatchSize, Train, T, decltype(trained)>(
          std::move(shuffled_training_data), std::forward<T>(learning_rate),
          trained);
    }
    return trained;
  }
  template <class Test>
  constexpr auto single_forward(Test &&x) {
    T new_layers;
    sprout::get<0>(new_layers) = sprout::get<0>(layers).make_by_input_data(x);
    forward_impl(new_layers, new_layers);
    return sprout::get<std::tuple_size_v<T> - 1>(new_layers).output_data;
  }

  template <class Test>
  constexpr auto evaluate(Test &&test_data) const {
    auto sum = 0;
    for (const auto &&[x, y] : test_data.get_data()) {
      if (single_forward(x).argmax() == y.argmax()) ++sum;
    }
    return sum;
  }
};
}  // namespace scenn
#endif
