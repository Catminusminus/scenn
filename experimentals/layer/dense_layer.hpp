#ifndef SCENN_EXPERIMENTALS_LAYER_DENSE_LAYER_HPP
#define SCENN_EXPERIMENTALS_LAYER_DENSE_LAYER_HPP
#include <scenn/matrix.hpp>
#include <scenn/util.hpp>
#include <utility>

namespace scenn {
namespace experimental {
template <std::size_t InputDim, std::size_t OutputDim, class NumType>
struct DenseLayer {
  decltype(make_zeros_from_pair<InputDim, 1, NumType>()) input_data;
  decltype(make_zeros_from_pair<OutputDim, 1, NumType>()) output_data;
  decltype(make_zeros_from_pair<InputDim, 1, NumType>()) input_delta;
  decltype(make_zeros_from_pair<InputDim, 1, NumType>()) output_delta;
  decltype(make_random_matrix<OutputDim, InputDim, NumType>()) weight;
  decltype(make_random_matrix<OutputDim, 1, NumType>()) bias;
  decltype(make_zeros_from_pair<OutputDim, InputDim, NumType>()) delta_W;
  decltype(make_zeros_from_pair<OutputDim, 1, NumType>()) delta_b;
  SCENN_CONSTEXPR DenseLayer(std::size_t seed)
      : input_data(make_zeros_from_pair<InputDim, 1, NumType>()),
        output_data(make_zeros_from_pair<OutputDim, 1, NumType>()),
        input_delta(make_zeros_from_pair<InputDim, 1, NumType>()),
        output_delta(make_zeros_from_pair<InputDim, 1, NumType>()),
        weight(make_random_matrix<OutputDim, InputDim, NumType>(1 + seed)),
        bias(make_random_matrix<OutputDim, 1, NumType>(3 + seed)),
        delta_w(make_zeros_from_pair<OutputDim, InputDim, NumType>()),
        delta_b(make_zeros_from_pair<OutputDim, 1, NumType>()){};
  template <class T>
  SCENN_CONSTEXPR auto forward(T&& data) {
    output_data = weight.dot(std::forward<T>(data)) + bias;
  }
  template <class T, class U>
  SCENN_CONSTEXPR auto backward(T&& data, U&& delta) {
    output_delta = weight.transposed().dot(delta);
    delta_w += delta.dot(std::forward<T>(data).transposed());
    delta_b += std::forward<U>(delta);
  }
  template <class T>
  SCENN_CONSTEXPR auto update_params(T&& rate) {
    weight -= delta_w * rate;
    bias -= delta_b * std::forward<T>(rate);
  }
  SCENN_CONSTEXPR auto clear_deltas() {
    delta_w = make_zeros_from_pair<OutputDim, InputDim, NumType>();
    delta_b = make_zeros_from_pair<OutputDim, 1, NumType>();
  }
};
}  // namespace experimental
}  // namespace scenn

#endif
