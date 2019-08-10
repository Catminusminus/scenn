#ifndef SCENN_LOAD_MINI_MNIST_HPP
#define SCENN_LOAD_MINI_MNIST_HPP

#include <sprout/tuple.hpp>
#include <scenn/util.hpp>
#include <scenn/dataset.hpp>

namespace scenn {
template <class NumType>
SCENN_CONSTEXPR auto load_mini_mnist_data() {
  SCENN_STATIC NumType x_train[100][784] = {
#include <tools/x_train>
  };
  SCENN_STATIC NumType x_test[10][784] = {
#include <tools/x_test>
  };
  SCENN_STATIC NumType y_train[100][3] = {
#include <tools/y_train>
  };
  SCENN_STATIC NumType y_test[10][3] = {
#include <tools/y_test>
  };
  return std::make_pair(Dataset(make_matrix_from_array(std::move(x_train)),
              make_matrix_from_array(std::move(y_train))), Dataset(make_matrix_from_array(std::move(x_test)),
              make_matrix_from_array(std::move(y_test))));
}
}  // namespace scenn

#endif
