#ifndef SCENN_LOSS_MSE_HPP
#define SCENN_LOSS_MSE_HPP

#include <scenn/util.hpp>

namespace scenn {
struct MSE {
  template <class T, class U>
  static SCENN_CONSTEXPR auto loss_function(const T& predictions,
                                            const U& labels) {
    auto diff = predictions - labels;
    return (diff.dot(diff.transposed()) * 0.5).to_value();
  }
  template <class T, class U>
  static SCENN_CONSTEXPR auto loss_derivative(const T& predictions,
                                              const U& labels) {
    return predictions - labels;
  }
};
}  // namespace scenn
#endif
