#ifndef SCENN_EXPERIMENTAL_LAYER_ACTIVATION_LAYER_HPP
#define SCENN_EXPERIMENTAL_LAYER_ACTIVATION_LAYER_HPP

#include <scenn/matrix.hpp>
#include <scenn/util.hpp>
#include <utility>

namespace scenn::experimental {
namespace detail {
// Activation must has an activation method and an activation_prime method
template <std::size_t Dim, class NumType, class Activation>
struct ActivationLayerImpl {
  Activation activation;
  decltype(make_zeros_from_pair<Dim, 1, NumType>()) input_data;
  decltype(make_zeros_from_pair<Dim, 1, NumType>()) output_data;
  decltype(make_zeros_from_pair<Dim, 1, NumType>()) input_delta;
  decltype(make_zeros_from_pair<Dim, 1, NumType>()) output_delta;
  SCENN_CONSTEXPR ActivationLayerImpl(Activation&& activation)
      : activation(std::forward<Activation>(activation)),
        input_data(make_zeros_from_pair<Dim, 1, NumType>()),
        output_data(make_zeros_from_pair<Dim, 1, NumType>()),
        input_delta(make_zeros_from_pair<Dim, 1, NumType>()),
        output_delta(make_zeros_from_pair<Dim, 1, NumType>()){};
  template <class T>
  SCENN_CONSTEXPR auto forward(T&& data) {
    output_data = activation.activate(std::forward<T>(data));
  }
  template <class T, class U>
  SCENN_CONSTEXPR auto backward(T&& data, U&& delta) {
    output_delta = activation.calc_backward_pass(std::forward<U>(data),
                                                 std::forward<T>(delta));
  }
  SCENN_CONSTEXPR auto clear_deltas() {}
  template <class T>
  SCENN_CONSTEXPR auto update_params([[maybe_unused]] T rate) {}
};
}  // namespace detail
template <std::size_t Dim, class NumType, class Activation>
SCENN_CONSTEXPR auto ActivationLayer(Activation&& activation) {
  return detail::ActivationLayerImpl<Dim, NumType, Activation>(
      std::forward<Activation>(activation));
}
}  // namespace scenn::experimental
#endif
