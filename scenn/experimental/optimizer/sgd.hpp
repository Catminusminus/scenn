#ifndef SCENN_EXPERIMENTAL_OPTIMIZER_SGD_HPP
#define SCENN_EXPERIMENTAL_OPTIMIZER_SGD_HPP

namespace scenn::experimental {
template <class T>
struct SGD {
  T lr;
  constexpr SGD(T lr=0.01): lr(lr) {};
  constexpr update() {};
};
}

#endif
