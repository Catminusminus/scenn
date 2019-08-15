#ifndef SCENN_LOSS_BINARY_CROSS_ENTROPY_HPP
#define SCENN_LOSS_BINARY_CROSS_ENTROPY_HPP

#include <scenn/matrix.hpp>
#include <scenn/util.hpp>
#include <sprout/math.hpp>

namespace scenn {
struct BinaryCrossEntropy {
  template <size_t N, class T, class U>
  static SCENN_CONSTEXPR auto loss_function(const Vector<N, T>& predictions,
                                            const Vector<N, U>& labels) {
    return -labels.dot(predictions.fmap(sprout::math::log<T>).transposed())
                .to_value() -
           ((labels * (-1) + 1.0)
                .dot((predictions * (-1) + 1.0)
                         .fmap(sprout::math::log<T>)
                         .transposed()))
               .to_value();
  }
  template <class T, class U>
  static SCENN_CONSTEXPR auto loss_derivative(const T& predictions,
                                              const U& labels) {
    return ((labels / predictions) * (-1)) +
           ((labels * (-1) + 1.0) / (predictions * (-1) + 1.0));
  }
};
}  // namespace scenn
#endif
