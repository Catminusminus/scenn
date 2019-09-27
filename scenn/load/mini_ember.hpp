#ifndef SCENN_LOAD_MINI_EMBER_HPP
#define SCENN_LOAD_MINI_EMBER_HPP

#include <scenn/dataset.hpp>
#include <scenn/util.hpp>
#include <sprout/tuple.hpp>

namespace scenn {
template <class NumType>
SCENN_CONSTEXPR auto load_mini_ember_data() {
  SCENN_STATIC NumType x_train[100][256] = {
#include <tools/x_train_ember>
  };
  SCENN_STATIC NumType x_test[10][256] = {
#include <tools/x_test_ember>
  };
  SCENN_STATIC NumType y_train[100][2] = {
#include <tools/y_train_ember>
  };
  SCENN_STATIC NumType y_test[10][2] = {
#include <tools/y_test_ember>
  };
  return std::make_pair(Dataset(make_matrix_from_array(std::move(x_train)),
                                make_matrix_from_array(std::move(y_train))),
                        Dataset(make_matrix_from_array(std::move(x_test)),
                                make_matrix_from_array(std::move(y_test))));
}
}  // namespace scenn

#endif