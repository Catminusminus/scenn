#ifndef SCENN_LAYER_ACTIVATION_LAYER_HPP
#define SCENN_LAYER_ACTIVATION_LAYER_HPP

namespace scenn {
template <class A, class B, class C, class D, class E>
struct ActivationLayer {
  A activation;
  B input_data;
  C output_data;
  D input_delta;
  E output_delta;
  std::size_t input_dim;
  std::size_t output_dim;
  constexpr ActivationLayer(A &&activation, std::size_t input_dim)
      : activate(std::forward<A>(activation)),
        input_data(std::nullopt),
        output_data(std::nullopt),
        input_delta(std::nullopt),
        output_delta(std::nullopt),
        input_dim(input_dim),
        output_dim(input_dim){};
  constexpr ActivationLayer(A activation; B input_data, C output_data,
                                          D input_delta, E output_delta,
                                          std::size_t input_dim,
                                          std::size_t output_dim)
      : activation(activation),
        input_data(input_data),
        output_data(output_data),
        input_delta(input_delta),
        output_delta(output_delta),
        input_dim(input_dim),
        output_dim(output_dim){};
  template <class T>
  static constexpr auto produce(T &&activation, std::size_t input_dim) const {
    return ActivationLayer(std::forward<T>(activation), std::nullopt,
                           std::nullopt, std::nullopt, std::nullopt, input_dim,
                           input_dim);
  };
  template <class T>
  constexpr auto forward(T &&data) const {
    return ActivationLayer<
        A, B, decltype(activation.activate(std::forward<T>(data))), D, E>(
        activation, input_data, activation.activate(std::forward<T>(data)),
        input_delta, output_delta, input_dim, output_dim);
  }
  template <class T, class U>
  constexpr auto backward(T &&data, U &&delta) const {
    return ActivationLayer<A, B, C, D,
                           decltype(
                               activation.activate_prime(std::forward<U>(data))
                                   .dot(std::forward<T>(delta)))>(
        activation, input_data, output_data, input_delta,
        activation.activate_prime(std::forward<U>(data))
            .dot(std::forward<T>(delta)),
        input_dim, output_dim);
  }
  constexpr auto clear_deltas() const { return (*this); }
  constexpr auto clear_deltas() && { return std::move(*this); }
  constexpr auto update_params() const { return (*this); }
  constexpr auto update_params() && { return std::move(*this); }
};
}  // namespace scenn
#endif
