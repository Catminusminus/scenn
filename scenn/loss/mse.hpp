#ifndef SCENN_LOSS_MSE_HPP
#define SCENN_LOSS_MSE_HPP

struct MSE {
  template <class T, class U>
  static constexpr auto loss_function(const T& preditions, const U& labels) {
    auto diff = preditions - labels;
    return (diff.dot(diff.transposed()) * 0.5).to_value();
  }
  template <class T, class U>
  static constexpr auto loss_derivative(const T& preditions, const U& labels) {
    return preditions - labels;
  }
};

#endif
