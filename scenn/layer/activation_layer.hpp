#ifndef SCENN_LAYER_ACTIVATION_LAYER_HPP
#define SCENN_LAYER_ACTIVATION_LAYER_HPP

#include <scenn/matrix.hpp>
#include <utility>

namespace scenn {
// A must has an activation method and an activation_prime method
template <class A, class B, class C, class D, class E>
struct ActivationLayerImpl {
  A activation;
  B input_data;
  C output_data;
  D input_delta;
  E output_delta;
  // constexpr ActivationLayerImpl(const ActivationLayerImpl& other) = default;
  // constexpr ActivationLayerImpl(ActivationLayerImpl&& other) = default;
  constexpr ActivationLayerImpl(A&& activation, B&& input_data, C&& output_data,
                                D&& input_delta, E&& output_delta)
      : activation(activation),
        input_data(input_data),
        output_data(output_data),
        input_delta(input_delta),
        output_delta(output_delta){};
  constexpr ActivationLayerImpl(const A& activation, const B& input_data,
                                const C& output_data, const D& input_delta,
                                const E& output_delta)
      : activation(activation),
        input_data(input_data),
        output_data(output_data),
        input_delta(input_delta),
        output_delta(output_delta){};
  template <class T>
  constexpr auto forward(T&& data) const& {
    return ActivationLayerImpl<
        A, B, decltype(activation.activate(std::forward<T>(data))), D, E>(
        activation, input_data, activation.activate(std::forward<T>(data)),
        input_delta, output_delta);
  }
  template <class T>
  constexpr auto forward(T&& data) && {
    return ActivationLayerImpl<
        A, B, decltype(activation.activate(std::forward<T>(data))), D, E>(
        std::move(activation), std::move(input_data),
        activation.activate(std::forward<T>(data)), std::move(input_delta),
        std::move(output_delta));
  }
  template <class T, class U>
  constexpr auto backward(T&& data, U&& delta) const& {
    return ActivationLayerImpl<A, B, C, D,
                               decltype(activation.activate_prime(
                                            std::forward<U>(data)) *
                                        (std::forward<T>(delta)))>(
        activation, input_data, output_data, input_delta,
        activation.activate_prime(std::forward<U>(data)) *
            (std::forward<T>(delta)));
  }
  template <class T, class U>
  constexpr auto backward(T&& data, U&& delta) && {
    return ActivationLayerImpl<
        A, B, C, D,
        decltype(activation.activate_prime(std::forward<U>(data))
                     *(std::forward<T>(delta)))>(
        std::move(activation), std::move(input_data), std::move(output_data),
        std::move(input_delta),
        activation.activate_prime(std::forward<U>(data))
            *(std::forward<T>(delta)));
  }
  template <class T>
  constexpr auto make_by_input_data(T&& input_data) const& {
    return ActivationLayerImpl<A, std::remove_reference_t<T>, C, D, E>(
        activation, std::forward<T>(input_data), output_data, input_delta,
        output_delta);
  }
  template <class T>
  constexpr auto make_by_input_data(T&& input_data) && {
    return ActivationLayerImpl<A, std::remove_reference_t<T>, C, D, E>(
        std::move(activation), std::forward<T>(input_data),
        std::move(output_data), std::move(input_delta),
        std::move(output_delta));
  }
  template <class T>
  constexpr auto make_by_input_delta(T&& input_delta) const& {
    return ActivationLayerImpl<A, B, C, std::remove_reference_t<T>, E>(
        activation, input_data, output_data, std::forward<T>(input_delta),
        output_delta);
  }
  template <class T>
  constexpr auto make_by_input_delta(T&& input_delta) && {
    return ActivationLayerImpl<A, B, C, std::remove_reference_t<T>, E>(
        std::move(activation), std::move(input_data), std::move(output_data),
        std::forward<T>(input_delta), std::move(output_delta));
  }
  constexpr auto clear_deltas() const& { return (*this); }
  constexpr auto clear_deltas() && { return std::move(*this); }
  template <class T>
  constexpr auto update_params([[maybe_unused]] T rate) const& { return (*this); }
  template <class T>
  constexpr auto update_params([[maybe_unused]] T rate) && { return std::move(*this); }
};

template <std::size_t Dim, class NumType, class Loss>
constexpr auto ActivationLayer(Loss&& loss) {
  return ActivationLayerImpl<Loss,
                             decltype(make_zeros_from_pair<Dim, 1, NumType>()),
                             decltype(make_zeros_from_pair<Dim, 1, NumType>()),
                             decltype(make_zeros_from_pair<Dim, 1, NumType>()),
                             decltype(make_zeros_from_pair<Dim, 1, NumType>())>(
      std::forward<Loss>(loss), make_zeros_from_pair<Dim, 1, NumType>(),
      make_zeros_from_pair<Dim, 1, NumType>(),
      make_zeros_from_pair<Dim, 1, NumType>(),
      make_zeros_from_pair<Dim, 1, NumType>());
}
}  // namespace scenn
#endif
