#ifndef SCENN_LOSS_CROSS_ENTROPY_HPP
#define SCENN_LOSS_CROSS_ENTROPY_HPP

#include <scenn/matrix.hpp>
#include <sprout/math.hpp>

namespace scenn {
struct CrossEntropy {
  template <size_t N, class T, class U>
  static constexpr auto loss_function(const Vector<N, T>& preditions,
                                      const Vector<N, U>& labels) {
    return -labels.dot(preditions.fmap(sprout::math::log<T>).transposed())
                .to_value();
  }
  template <class T, class U>
  static constexpr auto loss_derivative(const T& preditions, const U& labels) {
    return preditions - labels;
  }
};
}  // namespace scenn
#endif
