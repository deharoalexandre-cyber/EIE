// EIE — API Server Header
// Apache License 2.0
#pragma once
#include "../core/config.h"
#include "../core/scheduling.h"
#include "../core/model_manager.h"
#include "../monitoring/monitoring.h"

namespace eie {

void startServer(const ServerConfig& cfg, ModelManager& models,
                 GroupScheduler& scheduler, PolicyStrategy* policy,
                 Metrics& metrics, AuditTrail& audit);

} // namespace eie
