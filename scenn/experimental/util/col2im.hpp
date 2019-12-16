#include <scenn/experimental/tensor/tensor3d.hpp>
#include <scenn/experimental/util/pad.hpp>
#include <scenn/util.hpp>

namespace scenn::experimental {
template <std::size_t F, std::size_t O, std::size_t Stride, std::size_t Pad,
          std::size_t L, std::size_t M, std::size_t N, class T>
SCENN_CONSTEXPR auto col2im1d(const Matrix<M * F, L * O, T>& mat) {
  T cols[M][F][L][O] = {};
  for (std::size_t zero_i = 0; zero_i < M; ++zero_i)
    for (std::size_t one_i = 0; one_i < F; ++one_i)
      for (std::size_t two_i = 0; two_i < L; ++two_i)
        for (std::size_t three_i = 0; three_i < O; ++three_i)
          cols[zero_i][one_i][two_i][three_i] =
              mat(zero_i * F + one_i, two_i * O + three_i);

  T transposed_cols[L][M][F][O] = {};
  for (std::size_t zero_i = 0; zero_i < M; ++zero_i)
    for (std::size_t one_i = 0; one_i < F; ++one_i)
      for (std::size_t two_i = 0; two_i < L; ++two_i)
        for (std::size_t three_i = 0; three_i < O; ++three_i)
          transposed_cols[two_i][zero_i][one_i][three_i] =
              cols[zero_i][one_i][two_i][three_i];

  Tensor3D<L, M, N + 2 * Pad + Stride - 1, T> ret;
  for (std::size_t j = 0; j < F; ++j)
    for (std::size_t two_i = 0; two_i < O; ++two_i)
      for (std::size_t zero_i = 0; zero_i < L; ++zero_i)
        for (std::size_t one_i = 0; one_i < M; ++one_i)
          ret(zero_i, one_i, j + two_i * Stride) +=
              transposed_cols[zero_i][one_i][j][two_i];
  return ret.template slice<Pad, Pad + N, 2>();
}
}  // namespace scenn::experimental