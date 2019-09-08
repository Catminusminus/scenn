#ifndef SCENN_LAYER_DENSE_LAYER_HPP
#define SCENN_LAYER_DENSE_LAYER_HPP
#include <scenn/matrix.hpp>
#include <scenn/util.hpp>
#include <utility>

namespace scenn {
namespace detail {
template <class A, class B, class C, class D, class E, class F, class G,
          class H, std::size_t InputDim, std::size_t OutputDim, class NumType>
struct DenseLayerImpl {
  A input_data;
  B output_data;
  C input_delta;
  D output_delta;
  E weight;
  F bias;
  G delta_w;
  H delta_b;
  // SCENN_CONSTEXPR DenseLayerImpl(const DenseLayerImpl& other) = default;
  // SCENN_CONSTEXPR DenseLayerImpl(DenseLayerImpl&& other) = default;
  SCENN_CONSTEXPR DenseLayerImpl(A&& input_data, B&& output_data,
                                 C&& input_delta, D&& output_delta, E&& weight,
                                 F&& bias, G&& delta_w, H&& delta_b)
      : input_data(input_data),
        output_data(output_data),
        input_delta(input_delta),
        output_delta(output_delta),
        weight(weight),
        bias(bias),
        delta_w(delta_w),
        delta_b(delta_b){};
  SCENN_CONSTEXPR DenseLayerImpl(const A& input_data, const B& output_data,
                                 const C& input_delta, const D& output_delta,
                                 const E& weight, const F& bias,
                                 const G& delta_w, const H& delta_b)
      : input_data(input_data),
        output_data(output_data),
        input_delta(input_delta),
        output_delta(output_delta),
        weight(weight),
        bias(bias),
        delta_w(delta_w),
        delta_b(delta_b){};
  template <class T>
  SCENN_CONSTEXPR auto make_by_input_data(T&& input_data) const& {
    return DenseLayerImpl<std::remove_reference_t<T>, B, C, D, E, F, G, H,
                          InputDim, OutputDim, NumType>(
        std::forward<T>(input_data), output_data, input_delta, output_delta,
        weight, bias, delta_w, delta_b);
  }
  template <class T>
  SCENN_CONSTEXPR auto make_by_input_data(T&& input_data) && {
    return DenseLayerImpl<std::remove_reference_t<T>, B, C, D, E, F, G, H,
                          InputDim, OutputDim, NumType>(
        std::forward<T>(input_data), std::move(output_data),
        std::move(input_delta), std::move(output_delta), std::move(weight),
        std::move(bias), std::move(delta_w), std::move(delta_b));
  }
  template <class T>
  SCENN_CONSTEXPR auto make_by_input_delta(T&& input_delta) const& {
    return DenseLayerImpl<A, B, std::remove_reference_t<T>, D, E, F, G, H,
                          InputDim, OutputDim, NumType>(
        input_data, output_data, std::forward<T>(input_delta), output_delta,
        weight, bias, delta_w, delta_b);
  }
  template <class T>
  SCENN_CONSTEXPR auto make_by_input_delta(T&& input_delta) && {
    return DenseLayerImpl<A, B, std::remove_reference_t<T>, D, E, F, G, H,
                          InputDim, OutputDim, NumType>(
        std::move(input_data), std::move(output_data),
        std::forward<T>(input_delta), std::move(output_delta),
        std::move(weight), std::move(bias), std::move(delta_w),
        std::move(delta_b));
  }
  template <class T>
  SCENN_CONSTEXPR auto forward(T&& data) const& {
    return DenseLayerImpl<A, decltype(weight.dot(data) + bias), C, D, E, F, G,
                          H, InputDim, OutputDim, NumType>(
        input_data, weight.dot(std::forward<T>(data)) + bias, input_delta,
        output_delta, weight, bias, delta_w, delta_b);
  }
  template <class T>
  SCENN_CONSTEXPR auto forward(T&& data) && {
    auto new_output_data = weight.dot(std::forward<T>(data)) + bias;
    return DenseLayerImpl<A, decltype(new_output_data), C, D, E, F, G, H,
                          InputDim, OutputDim, NumType>(
        std::move(input_data), std::move(new_output_data),
        std::move(input_delta), std::move(output_delta), std::move(weight),
        std::move(bias), std::move(delta_w), std::move(delta_b));
  }
  template <class T, class U>
  SCENN_CONSTEXPR auto backward(T&& data, U&& delta) const& {
    return DenseLayerImpl<A, B, C, decltype(weight.transposed().dot(delta)), E,
                          F, decltype(delta_w + delta.dot(data.transposed())),
                          decltype(delta_b + delta), InputDim, OutputDim,
                          NumType>(
        input_data, output_data, input_delta, weight.transposed().dot(delta),
        weight, bias, delta_w + delta.dot(data.transposed()), delta_b + delta);
  }
  template <class T, class U>
  SCENN_CONSTEXPR auto backward(T&& data, U&& delta) && {
    auto new_output_delta = weight.transposed().dot(delta);
    auto new_delta_w =
        std::move(delta_w) + delta.dot(std::forward<T>(data).transposed());
    auto new_delta_b = std::move(delta_b) + std::forward<U>(delta);
    return DenseLayerImpl<A, B, C, decltype(new_output_delta), E, F,
                          decltype(new_delta_w), decltype(new_delta_b),
                          InputDim, OutputDim, NumType>(
        input_data, output_data, input_delta, std::move(new_output_delta),
        weight, bias, std::move(new_delta_w), std::move(new_delta_b));
  }
  template <class T>
  SCENN_CONSTEXPR auto update_params(T&& rate) const& {
    return DenseLayerImpl<A, B, C, D, decltype(weight - delta_w * rate),
                          decltype(bias - delta_b * rate), G, H, InputDim,
                          OutputDim, NumType>(
        input_data, output_data, input_delta, output_delta,
        weight - delta_w * rate, bias - delta_b * rate, delta_w, delta_b);
  }
  template <class T>
  SCENN_CONSTEXPR auto update_params(T&& rate) && {
    auto new_weight = weight - delta_w * rate;
    auto new_bias = bias - delta_b * std::forward<T>(rate);
    return DenseLayerImpl<A, B, C, D, decltype(weight - delta_w * rate),
                          decltype(bias - delta_b * rate), G, H, InputDim,
                          OutputDim, NumType>(
        std::move(input_data), std::move(output_data), std::move(input_delta),
        std::move(output_delta), std::move(new_weight), std::move(new_bias),
        std::move(delta_w), std::move(delta_b));
  }
  SCENN_CONSTEXPR auto clear_deltas() const& {
    return DenseLayerImpl<
        A, B, C, D, E, F,
        decltype(make_zeros_from_pair<OutputDim, InputDim, NumType>()),
        decltype(make_zeros_from_pair<OutputDim, 1, NumType>()), InputDim,
        OutputDim, NumType>(
        input_data, output_data, input_delta, output_delta, weight, bias,
        make_zeros_from_pair<OutputDim, InputDim, NumType>(),
        make_zeros_from_pair<OutputDim, 1, NumType>());
  }
  SCENN_CONSTEXPR auto clear_deltas() && {
    return DenseLayerImpl<
        A, B, C, D, E, F,
        decltype(make_zeros_from_pair<OutputDim, InputDim, NumType>()),
        decltype(make_zeros_from_pair<OutputDim, 1, NumType>()), InputDim,
        OutputDim, NumType>(
        std::move(input_data), std::move(output_data), std::move(input_delta),
        std::move(output_delta), std::move(weight), std::move(bias),
        make_zeros_from_pair<OutputDim, InputDim, NumType>(),
        make_zeros_from_pair<OutputDim, 1, NumType>());
  }
};
}  // namespace detail
template <std::size_t InputDim, std::size_t OutputDim, class NumType>
SCENN_CONSTEXPR auto DenseLayer(std::size_t seed = 0) {
  return detail::DenseLayerImpl<
      decltype(make_zeros_from_pair<InputDim, 1, NumType>()),
      decltype(make_zeros_from_pair<OutputDim, 1, NumType>()),
      decltype(make_zeros_from_pair<InputDim, 1, NumType>()),
      decltype(make_zeros_from_pair<InputDim, 1, NumType>()),
      decltype(make_random_matrix<OutputDim, InputDim, NumType>()),
      decltype(make_random_matrix<OutputDim, 1, NumType>()),
      decltype(make_zeros_from_pair<OutputDim, InputDim, NumType>()),
      decltype(make_zeros_from_pair<OutputDim, 1, NumType>()), InputDim,
      OutputDim, NumType>(
      make_zeros_from_pair<InputDim, 1, NumType>(),
      make_zeros_from_pair<OutputDim, 1, NumType>(),
      make_zeros_from_pair<InputDim, 1, NumType>(),
      make_zeros_from_pair<InputDim, 1, NumType>(),
      make_random_matrix<OutputDim, InputDim, NumType>(1 + seed),
      make_random_matrix<OutputDim, 1, NumType>(3 + seed),
      make_zeros_from_pair<OutputDim, InputDim, NumType>(),
      make_zeros_from_pair<OutputDim, 1, NumType>());
}

template <std::size_t InputDim, std::size_t OutputDim, class NumType>
SCENN_CONSTEXPR auto DenseLayerCreator(std::size_t seed = 0) {
  return DenseLayer<InputDim, OutputDim, NumType>(seed);
}

}  // namespace scenn

#endif
