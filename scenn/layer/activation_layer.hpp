#ifndef SCENN_LAYER_ACTIVATION_LAYER_HPP
#define SCENN_LAYER_ACTIVATION_LAYER_HPP

#include <scenn/matrix.hpp>
#include <scenn/util.hpp>
#include <utility>

namespace scenn {
namespace detail {
// A must has an activation method and an activation_prime method
template <class A, class B, class C, class D, class E>
struct ActivationLayerImpl {
  A activation;
  B input_data;
  C output_data;
  D input_delta;
  E output_delta;
  // SCENN_CONSTEXPR ActivationLayerImpl(const ActivationLayerImpl& other) =
  // default; SCENN_CONSTEXPR ActivationLayerImpl(ActivationLayerImpl&& other) =
  // default;
  SCENN_CONSTEXPR ActivationLayerImpl(A&& activation, B&& input_data,
                                      C&& output_data, D&& input_delta,
                                      E&& output_delta)
      : activation(activation),
        input_data(input_data),
        output_data(output_data),
        input_delta(input_delta),
        output_delta(output_delta){};
  SCENN_CONSTEXPR ActivationLayerImpl(const A& activation, const B& input_data,
                                      const C& output_data,
                                      const D& input_delta,
                                      const E& output_delta)
      : activation(activation),
        input_data(input_data),
        output_data(output_data),
        input_delta(input_delta),
        output_delta(output_delta){};
  template <class T>
  SCENN_CONSTEXPR auto forward(T&& data) const& {
    return ActivationLayerImpl<
        A, B, decltype(activation.activate(std::forward<T>(data))), D, E>(
        activation, input_data, activation.activate(std::forward<T>(data)),
        input_delta, output_delta);
  }
  template <class T>
  SCENN_CONSTEXPR auto forward(T&& data) && {
    return ActivationLayerImpl<
        A, B, decltype(activation.activate(std::forward<T>(data))), D, E>(
        std::move(activation), std::move(input_data),
        activation.activate(std::forward<T>(data)), std::move(input_delta),
        std::move(output_delta));
  }
  template <class T, class U>
  SCENN_CONSTEXPR auto backward(T&& data, U&& delta) const& {
    return ActivationLayerImpl<A, B, C, D,
                               decltype(activation.calc_backward_pass(
                                   std::forward<U>(data),
                                   std::forward<T>(delta)))>(
        activation, input_data, output_data, input_delta,
        activation.calc_backward_pass(std::forward<U>(data),
                                      std::forward<T>(delta)));
  }
  template <class T, class U>
  SCENN_CONSTEXPR auto backward(T&& data, U&& delta) && {
    return ActivationLayerImpl<A, B, C, D,
                               decltype(activation.calc_backward_pass(
                                   std::forward<U>(data),
                                   std::forward<T>(delta)))>(
        std::move(activation), std::move(input_data), std::move(output_data),
        std::move(input_delta),
        activation.calc_backward_pass(std::forward<U>(data),
                                      std::forward<T>(delta)));
  }
  template <class T>
  SCENN_CONSTEXPR auto make_by_input_data(T&& input_data) const& {
    return ActivationLayerImpl<A, std::remove_reference_t<T>, C, D, E>(
        activation, std::forward<T>(input_data), output_data, input_delta,
        output_delta);
  }
  template <class T>
  SCENN_CONSTEXPR auto make_by_input_data(T&& input_data) && {
    return ActivationLayerImpl<A, std::remove_reference_t<T>, C, D, E>(
        std::move(activation), std::forward<T>(input_data),
        std::move(output_data), std::move(input_delta),
        std::move(output_delta));
  }
  template <class T>
  SCENN_CONSTEXPR auto make_by_input_delta(T&& input_delta) const& {
    return ActivationLayerImpl<A, B, C, std::remove_reference_t<T>, E>(
        activation, input_data, output_data, std::forward<T>(input_delta),
        output_delta);
  }
  template <class T>
  SCENN_CONSTEXPR auto make_by_input_delta(T&& input_delta) && {
    return ActivationLayerImpl<A, B, C, std::remove_reference_t<T>, E>(
        std::move(activation), std::move(input_data), std::move(output_data),
        std::forward<T>(input_delta), std::move(output_delta));
  }
  SCENN_CONSTEXPR auto clear_deltas() const& { return (*this); }
  SCENN_CONSTEXPR auto clear_deltas() && { return std::move(*this); }
  template <class T>
  SCENN_CONSTEXPR auto update_params([[maybe_unused]] T rate) const& {
    return (*this);
  }
  template <class T>
  SCENN_CONSTEXPR auto update_params([[maybe_unused]] T rate) && {
    return std::move(*this);
  }
};
}  // namespace detail
template <std::size_t Dim, class NumType, class Loss>
SCENN_CONSTEXPR auto ActivationLayer(Loss&& loss) {
  return detail::ActivationLayerImpl<
      Loss, decltype(make_zeros_from_pair<Dim, 1, NumType>()),
      decltype(make_zeros_from_pair<Dim, 1, NumType>()),
      decltype(make_zeros_from_pair<Dim, 1, NumType>()),
      decltype(make_zeros_from_pair<Dim, 1, NumType>())>(
      std::forward<Loss>(loss), make_zeros_from_pair<Dim, 1, NumType>(),
      make_zeros_from_pair<Dim, 1, NumType>(),
      make_zeros_from_pair<Dim, 1, NumType>(),
      make_zeros_from_pair<Dim, 1, NumType>());
}

template <std::size_t Dim, class NumType, class Loss>
SCENN_CONSTEXPR auto ActivationLayerCreator(Loss&& loss) {
  return ActivationLayer<Dim, NumType, Loss>(std::forward<Loss>(loss));
}

}  // namespace scenn
#endif
