#ifndef SCENN_MODEL_SEQUENTIAL_NETWORK_HPP
#define SCENN_MODEL_SEQUENTIAL_NETWORK_HPP

#include <scenn/util.hpp>
#include <sprout/tuple.hpp>

namespace scenn {
template <class LossFunction, class... Layers>
struct SequentialNetwork {
  LossFunction loss;
  using T = sprout::tuple<Layers...>;
  T layers;
  SCENN_CONSTEXPR SequentialNetwork(LossFunction &&loss, Layers &&... layers)
      : loss(loss), layers(sprout::make_tuple(layers...)){};
  SCENN_CONSTEXPR SequentialNetwork(const LossFunction &loss,
                                    const Layers &... layers)
      : loss(loss), layers(sprout::make_tuple(layers...)){};
  template <size_t index>
  SCENN_CONSTEXPR auto get_forward_input(const T& some_layers) const {
    if constexpr (index == 0) {
      return sprout::get<index>(some_layers).input_data;
    } else {
      return sprout::get<index - 1>(some_layers).output_data;
    }
  }
  template <size_t index>
  SCENN_CONSTEXPR auto get_backward_input(const T& some_layers) const {
    if constexpr (index + 1 == std::tuple_size_v<T>) {
      return sprout::get<index>(some_layers).input_delta;
    } else {
      return sprout::get<index + 1>(some_layers).output_delta;
    }
  }

  template <std::size_t index = 0, class I>
  SCENN_CONSTEXPR auto update_impl_impl(T &new_layers, I rate) const {
    if constexpr (index < std::tuple_size_v<T>) {
      sprout::get<index>(new_layers) =
          sprout::get<index>(layers).update_params(rate);
      update_impl_impl<index + 1>(new_layers, rate);
    }
  }

  template <class I>
  SCENN_CONSTEXPR auto update_params(T &new_layers, I rate) const {
    update_impl_impl(new_layers, rate);
  }

  template <std::size_t index = 0>
  SCENN_CONSTEXPR auto clear_impl_impl(T &new_layers, T &old_layers) const {
    if constexpr (index < std::tuple_size_v<T>) {
      sprout::get<index>(new_layers) =
          sprout::get<index>(old_layers).clear_deltas();
      clear_impl_impl<index + 1>(new_layers, old_layers);
    }
  }
  SCENN_CONSTEXPR auto clear_deltas(T &new_layers, T &old_layers) const {
    clear_impl_impl(new_layers, old_layers);
  }
  template <class Train, class I>
  SCENN_CONSTEXPR auto update(Train &&mini_batch, I &&learning_rate) const {
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
  SCENN_CONSTEXPR auto forward_impl(T &new_layers, T &old_layers) const {
    if constexpr (index < std::tuple_size_v<T>) {
      sprout::get<index>(new_layers) =
          sprout::get<index>(old_layers).forward(get_forward_input<index>(old_layers));
      forward_impl<index + 1>(new_layers, old_layers);
    }
  }

  template <std::size_t index = std::tuple_size_v<T> - 1>
  SCENN_CONSTEXPR auto backward_impl(T &new_layers, T &old_layers) const {
    sprout::get<index>(new_layers) =
        sprout::get<index>(old_layers)
            .backward(get_forward_input<index>(old_layers), get_backward_input<index>(old_layers));
    if constexpr (index > 0) {
      backward_impl<index - 1>(new_layers, old_layers);
    }
  }

  template <class Train>
  SCENN_CONSTEXPR auto forward_backward(Train &&mini_batch) const {
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
#ifdef SCENN_DISABLE_CONSTEXPR
      std::cout << loss.loss_function(
                       single_forward(x.transposed()).transposed(), y)
                << std::endl;
#endif
    }
    return sprout::make_from_tuple<SequentialNetwork>(
        sprout::tuple_cat(sprout::tie(loss), new_layers));
  }

  template <class Train, class I>
  SCENN_CONSTEXPR auto train_batch(Train &&mini_batch, I learning_rate) {
    return forward_backward(mini_batch)
        .update(std::forward<Train>(mini_batch), learning_rate);
  }

  template <std::size_t Index = 0, std::size_t Upper, std::size_t Interval,
            class Train, class T, class Model>
  SCENN_CONSTEXPR auto train_impl(Train &&training_data, T &&learning_rate,
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
  SCENN_CONSTEXPR auto train(Train &&training_data, std::size_t epochs,
                             T &&learning_rate) {
    constexpr auto N = std::remove_reference_t<Train>::length();
    auto trained = (*this);
    for (std::size_t epoch = 0; epoch < epochs; ++epoch) {
      auto shuffled_training_data = training_data.shuffle(epoch);
      train_impl<0, N, MiniBatchSize, std::remove_reference_t<Train>&&, T, decltype(trained)>(
          std::move(shuffled_training_data), std::forward<T>(learning_rate),
          trained);
    }
    return trained;
  }
  template <class Test>
  SCENN_CONSTEXPR auto single_forward(Test &&x) const {
    // TODO CHECK!
    T new_layers = layers;
    sprout::get<0>(new_layers) =
        sprout::get<0>(layers).make_by_input_data(std::forward<Test>(x));
    forward_impl(new_layers, new_layers);
    return sprout::get<std::tuple_size_v<T> - 1>(new_layers).output_data;
  }

  template <class Test>
  SCENN_CONSTEXPR auto evaluate(Test &&test_data) const {
    auto sum = 0;
    for (auto &&[x, y] : test_data.get_data()) {
      if (single_forward(x.transposed()).transposed().argmax() == y.argmax())
        ++sum;
    }
    return sum;
  }
};
}  // namespace scenn
#endif
