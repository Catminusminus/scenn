#ifndef SCENN_LOAD_MINI_FIZZBUZZ_HPP
#define SCENN_LOAD_MINI_FIZZBUZZ_HPP

#include <scenn/dataset.hpp>
#include <scenn/util.hpp>
#include <sprout/array.hpp>

namespace scenn {
template <class NumType>
SCENN_CONSTEXPR auto load_mini_fizzbuzz_data() {
  SCENN_STATIC NumType x_train[1000][19] = {
#include <tools/x_train_fizzbuzz>
  };
  SCENN_STATIC NumType x_test[30][19] = {
#include <tools/x_test_fizzbuzz>
  };
  SCENN_STATIC NumType y_train[1000][4] = {
#include <tools/y_train_fizzbuzz>
  };
  SCENN_STATIC NumType y_test[30][4] = {
#include <tools/y_test_fizzbuzz>
  };
  sprout::array<NumType, 1000> x_train_orig = {
#include <tools/train_fizzbuzz_orig>
  };
  sprout::array<NumType, 30> x_test_orig = {
#include <tools/test_fizzbuzz_orig>
  };
  return std::make_tuple(Dataset(make_matrix_from_array(std::move(x_train)),
                                make_matrix_from_array(std::move(y_train))),
                        Dataset(make_matrix_from_array(std::move(x_test)),
                                make_matrix_from_array(std::move(y_test))),
                        std::pair(x_train_orig, x_test_orig)
                        );
}
}  // namespace scenn

#endif