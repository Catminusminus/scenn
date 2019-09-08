#ifndef SCENN_LAYER_ACTIVATION_LAYER_HPP
#define SCENN_LAYER_ACTIVATION_LAYER_HPP

#include <scenn/matrix.hpp>
#include <scenn/util.hpp>
#include <utility>

namespace scenn {
namespace experimental {
// Activation must has an activation method and an activation_prime method
template <class Dim, class NumType, class Activation>
struct ActivationLayer {
  Activation activation;
  decltype(make_zeros_from_pair<Dim, 1, NumType>()) input_data;
  decltype(make_zeros_from_pair<Dim, 1, NumType>()) output_data;
  decltype(make_zeros_from_pair<Dim, 1, NumType>()) input_delta;
  decltype(make_zeros_from_pair<Dim, 1, NumType>()) output_delta;
  SCENN_CONSTEXPR ActivationLayer(Activation&& activation)
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
}  // namespace experimental
}  // namespace scenn
#endif
