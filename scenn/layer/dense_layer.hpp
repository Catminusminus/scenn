#ifndef SCENN_LAYER_DENSE_LAYER_HPP
#define SCENN_LAYER_DENSE_LAYER_HPP

namespace scenn {
template <class B, class C, class D, class E, class H, class I, class J,
          class K>
struct DenseLayer {
  B input_data;
  C output_data;
  D input_delta;
  E output_delta;
  std::size_t input_dim;
  std::size_t output_dim;
  H weight;
  I bias;
  J delta_w;
  K delta_b;
  constexpr DenseLayer(B input_data, C output_data, D input_delta,
                       E output_delta, std::size_t input_dim,
                       std::size_t output_dim, H weight;
                       I bias; J delta_w; K delta_b;)
      : input_data(input_data),
        output_data(output_data),
        input_delta(input_delta),
        output_delta(output_delta),
        input_dim(input_dim),
        output_dim(output_dim),
        weight(weight),
        bias(bias),
        delta_w(delta_w),
        delta_b(delta_b){};
  template <class T, class U, size_t input_dim, size_t output_dim>
  static constexpr auto produce() const {
    return DenseLayer(std::nullopt, std::nullopt, std::nullopt, std::nullopt,
                      input_dim, output_dim,
                      make_random_matrix<output_dim, input_dim>(),
                      make_random_matrix<output_dim, 1>(),
                      make_zeros_from_pair<output_dim, input_dim>(),
                      make_zeros_from_pair<output_dim, 1>());
  }
  template <class T>
  constexpr auto make_by_input_data(T &&input_data) const {
    return DenseLayer(std::forward<T>(input_data), ...);
  }
  template <class T>
  constexpr auto make_by_input_delta(T &&input_delta) const {
    return DenseLayer(...);
  }
  template <class T>
  constexpr auto forward(T &&data) const {
    return DenseLayer(input_data, sum(weight.dot(std::forward<T>(data)), bias),
                      input_delta, output_delta, input_dim, output_dim, weight,
                      bias, delta_w, delta_b);
  }
  template <class T, class U>
  constexpr auto backward(T &&data, U &&delta) const {
    return DenseLayer(input_data, output_data, input_delta,
                      weight.transposed().dot(delta), input_dim, output_dim,
                      weight, bias, delta_w + delta.dot(data.transpose()),
                      delta_b + delta);
  }
  template <class T>
  constexpr auto update_params(T &&rate) const {
    return DenseLayer(input_data, output_data, input_delta, output_delta,
                      input_dim, output_dim, weight - delta_w * rate,
                      bias - delta_b * rate, delta_w, delta_b);
  }
  constexpr auto clear_deltas() const {
    return DenseLayer(input_data, output_data, input_delta, output_delta,
                      input_dim, output_dim, weight, bias,
                      make_zeros_from_pair<output_dim, input_dim>(),
                      make_zeros_from_pair<output_dim, 1>());
  }
};
}  // namespace scenn

#endif
