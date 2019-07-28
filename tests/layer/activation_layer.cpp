#include <scenn/activation.hpp>
#include <scenn/layer/activation_layer.hpp>

int main() {
  using namespace scenn;
  constexpr auto sigmoid_activation_layer = ActivationLayer(Sigmoid());
  constexpr auto deltas_cleared_updated =
      sigmoid_activation_layer.make_by_input_data(1)
          .make_by_input_delta(1)
          .clear_deltas()
          .update_params();
}
