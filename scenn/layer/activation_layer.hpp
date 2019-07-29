#ifndef SCENN_LAYER_ACTIVATION_LAYER_HPP
#define SCENN_LAYER_ACTIVATION_LAYER_HPP

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
  constexpr ActivationLayerImpl(const ActivationLayerImpl& other) = default;
  constexpr ActivationLayerImpl(ActivationLayerImpl&& other) = default;
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
    return ActivationLayerImpl<
        A, B, C, D,
        decltype(activation.activate_prime(std::forward<U>(data))
                     .dot(std::forward<T>(delta)))>(
        activation, input_data, output_data, input_delta,
        activation.activate_prime(std::forward<U>(data))
            .dot(std::forward<T>(delta)));
  }
  template <class T, class U>
  constexpr auto backward(T&& data, U&& delta) && {
    return ActivationLayerImpl<
        A, B, C, D,
        decltype(activation.activate_prime(std::forward<U>(data))
                     .dot(std::forward<T>(delta)))>(
        std::move(activation), std::move(input_data), std::move(output_data),
        std::move(input_delta),
        activation.activate_prime(std::forward<U>(data))
            .dot(std::forward<T>(delta)));
  }
  template <class T>
  constexpr auto make_by_input_data(T&& input_data) const& {
    return ActivationLayerImpl<A, T, C, D, E>(
        activation, std::forward<T>(input_data), output_data, input_delta,
        output_delta);
  }
  template <class T>
  constexpr auto make_by_input_data(T&& input_data) && {
    return ActivationLayerImpl<A, T, C, D, E>(
        std::move(activation), std::forward<T>(input_data),
        std::move(output_data), std::move(input_delta),
        std::move(output_delta));
  }
  template <class T>
  constexpr auto make_by_input_delta(T&& input_delta) const& {
    return ActivationLayerImpl<A, B, C, T, E>(
        activation, input_data, output_data, std::forward<T>(input_delta),
        output_delta);
  }
  template <class T>
  constexpr auto make_by_input_delta(T&& input_delta) && {
    return ActivationLayerImpl<A, B, C, T, E>(
        std::move(activation), std::move(input_data), std::move(output_data),
        std::forward<T>(input_delta), std::move(output_delta));
  }
  constexpr auto clear_deltas() const& { return (*this); }
  constexpr auto clear_deltas() && { return std::move(*this); }
  constexpr auto update_params() const& { return (*this); }
  constexpr auto update_params() && { return std::move(*this); }
};

template <class Loss>
constexpr auto ActivationLayer(Loss&& loss) {
  return ActivationLayerImpl<Loss, int, int, int, int>(std::forward<Loss>(loss),
                                                       1, 1, 1, 1);
}
}  // namespace scenn
#endif
