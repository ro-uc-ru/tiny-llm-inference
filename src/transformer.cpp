#include <tiny_llm/transformer.hpp>

namespace tiny_llm {

Transformer::Transformer(const std::filesystem::path& checkpoint_path)
    : file_{checkpoint_path}, config_{parse_config(file_.bytes())},
      weights_{map_weights(file_.bytes(), config_)}, state_{make_run_state(config_)} {}

} // namespace tiny_llm