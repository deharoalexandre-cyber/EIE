// PrismML runtime deliberately has no EIE expert-weight streaming port.
// CpuBackend rejects nonzero ews_slots before constructing this type.
#include "expert_stream.h"
#include <stdexcept>

namespace eie {
struct ExpertStream::Impl {};

ExpertStream::ExpertStream(const std::string &, int) {
    throw std::logic_error("EWS is unavailable with the PrismML llama.cpp runtime");
}
ExpertStream::~ExpertStream() = default;

void ExpertStream::bind(llama_model *) { throw std::logic_error("EWS unavailable"); }
void ExpertStream::configure(llama_context_params &) { throw std::logic_error("EWS unavailable"); }
const std::string & ExpertStream::error() const { throw std::logic_error("EWS unavailable"); }
ExpertStreamStats ExpertStream::stats() const { throw std::logic_error("EWS unavailable"); }
void ExpertStream::beginTrace(bool) { throw std::logic_error("EWS unavailable"); }
void ExpertStream::tracePhase(RoutingPhase) { throw std::logic_error("EWS unavailable"); }
void ExpertStream::endTrace(const std::string &) { throw std::logic_error("EWS unavailable"); }
RoutingHistogram ExpertStream::routing() const { throw std::logic_error("EWS unavailable"); }
bool ExpertStream::callback(ggml_tensor *, bool, void *) { throw std::logic_error("EWS unavailable"); }
} // namespace eie
