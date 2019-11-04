#ifndef SCENN_EXPERIMENTAL_UTIL_IM2COL_HPP
#define SCENN_EXPERIMENTAL_UTIL_IM2COL_HPP

#include <scenn/experimental/tensor/tensor3d.hpp>
#include <scenn/experimental/util/pad.hpp>
#include <scenn/util.hpp>

namespace scenn::experimental {
template <std::size_t F, std::size_t O, std::size_t Stride, std::size_t Pad,
          std::size_t L, std::size_t M, std::size_t N, class T>
SCENN_CONSTEXPR auto im2col1d(const Tensor3D<L, M, N, T>& tensor) {
  static_assert(O == (M - F + 2 * Pad) / Stride + 1);
  const auto pad_tensor = pad<Pad>(tensor);
  // constexpr auto O = (M - F + 2 * Pad) / Stride + 1;
  Matrix<N * F, L * O, T> ret;

  // Todo: implement constexpr N-dim Tensor and use it
  T data[L][N][F][O] = {};
  for (std::size_t j = 0; j < F; ++j)
    for (std::size_t zero_i = 0; zero_i < L; ++zero_i)
      for (std::size_t one_i = 0; one_i < N; ++one_i)
        for (std::size_t three_i = 0; three_i < O; ++three_i)
          data[zero_i][one_i][j][three_i] =
              pad_tensor(zero_i, j + three_i * Stride, one_i);

  T transposed_data[N][F][L][O] = {};
  for (std::size_t zero_i = 0; zero_i < L; ++zero_i)
    for (std::size_t one_i = 0; one_i < N; ++one_i)
      for (std::size_t three_i = 0; three_i < F; ++three_i)
        for (std::size_t four_i = 0; four_i < O; ++four_i)
          transposed_data[one_i][three_i][zero_i][four_i] =
              data[zero_i][one_i][three_i][four_i];

  for (std::size_t zero_i = 0; zero_i < N; ++zero_i)
    for (std::size_t one_i = 0; one_i < F; ++one_i)
      for (std::size_t two_i = 0; two_i < L; ++two_i)
        for (std::size_t three_i = 0; three_i < O; ++three_i)
          ret(zero_i * N + one_i, two_i * L + three_i) =
              transposed_data[zero_i][one_i][two_i][three_i];
  return ret;
}
}  // namespace scenn::experimental

#endif
