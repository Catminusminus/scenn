#ifndef SCENN_LAYER_DENSE_LAYER_HPP
#define SCENN_LAYER_DENSE_LAYER_HPP

#include <scenn/matrix.hpp>
#include <utility>

namespace scenn {
template <class B, class C, class D, class E, class H, class I, class J,
          class K, std::size_t input_dim, std::size_t output_dim, class NumType>
struct DenseLayerImpl {
  B input_data;
  C output_data;
  D input_delta;
  E output_delta;
  H weight;
  I bias;
  J delta_w;
  K delta_b;
  constexpr DenseLayerImpl(const DenseLayerImpl& other) = default;
  constexpr DenseLayerImpl(DenseLayerImpl&& other) = default;
  constexpr DenseLayerImpl(B&& input_data, C&& output_data, D&& input_delta,
                           E&& output_delta, H&& weight, I&& bias, J&& delta_w,
                           K&& delta_b)
      : input_data(std::forward<B>(input_data)),
        output_data(std::forward<C>(output_data)),
        input_delta(std::forward<D>(input_delta)),
        output_delta(std::forward<E>(output_delta)),
        weight(std::forward<H>(weight)),
        bias(std::forward<I>(bias)),
        delta_w(std::forward<J>(delta_w)),
        delta_b(std::forward<K>(delta_b)){};
  constexpr DenseLayerImpl(const B& input_data, const C& output_data,
                           const D& input_delta, const E& output_delta,
                           const H& weight, const I& bias, const J& delta_w,
                           const K& delta_b)
      : input_data(input_data),
        output_data(output_data),
        input_delta(input_delta),
        output_delta(output_delta),
        weight(weight),
        bias(bias),
        delta_w(delta_w),
        delta_b(delta_b){};
  template <class T>
  constexpr auto make_by_input_data(T&& input_data) const {
    return DenseLayerImpl(std::forward<T>(input_data), ...);
  }
  template <class T>
  constexpr auto make_by_input_delta(T&& input_delta) const {
    return DenseLayerImpl(...);
  }
  template <class T>
  constexpr auto forward(T&& data) const {
    return DenseLayerImpl(
        input_data, sum(weight.dot(std::forward<T>(data)), bias), input_delta,
        output_delta, weight, bias, delta_w, delta_b);
  }
  template <class T, class U>
  constexpr auto backward(T&& data, U&& delta) const {
    return DenseLayerImpl(
        input_data, output_data, input_delta, weight.transposed().dot(delta),
        weight, bias, delta_w + delta.dot(data.transpose()), delta_b + delta);
  }
  template <class T>
  constexpr auto update_params(T&& rate) const {
    return DenseLayerImpl(input_data, output_data, input_delta, output_delta,
                          weight - delta_w * rate, bias - delta_b * rate,
                          delta_w, delta_b);
  }
  constexpr auto clear_deltas() const {
    return DenseLayerImpl(
        input_data, output_data, input_delta, output_delta, weight, bias,
        make_zeros_from_pair<output_dim, input_dim, NumType>(),
        make_zeros_from_pair<output_dim, 1, Numtype>());
  }
};

template <std::size_t InputDim, std::size_t OutputDim, class NumType>
constexpr auto DenseLayer() {
  return DenseLayerImpl<
      int, int, int, int,
      decltype(make_random_matrix<output_dim, input_dim, NumType>()),
      decltype(make_random_matrix<output_dim, 1, NumType>()),
      decltype(make_zeros_from_pair<output_dim, input_dim, NumType>()),
      decltype(make_zeros_from_pair<output_dim, 1, NumType>()), InputDim,
      OutputDim, NumType>(
      1, 1, 1, 1, make_random_matrix<output_dim, input_dim, NumType>(),
      make_random_matrix<output_dim, 1, NumType>(),
      make_zeros_from_pair<output_dim, input_dim, NumType>(),
      make_zeros_from_pair<output_dim, 1, NumType>());
}
}  // namespace scenn

#endif
