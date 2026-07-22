/**
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <string>
#include <unordered_map>

namespace rapidsmpf::config {

/**
 * @brief String-form default values for config options.
 *
 * Defaults are stored as strings and parsed through the same factories used
 * for user-supplied values. `Options::get<T>(key, factory)` consults this
 * map automatically: when the user has not supplied a value, the factory
 * receives the registered default string for `key`.
 *
 * Options are not required to have an entry in this map. If no default is
 * registered for a key, the factory receives the empty string.
 *
 * To add a new option, add an entry here and reference it at the call site
 * via `Options::get<T>("<key>", factory)`.
 */
inline const std::unordered_map<std::string, std::string> DEFAULTS{
    {"statistics", "false"},
    {"pinned_memory", "false"},
    {"pinned_initial_pool_size", "0%"},
    {"pinned_max_pool_size", "80%"},
    {"spill_device_limit", "80%"},
    {"periodic_spill_check", "1ms"},
    {"num_streams", "16"},
    {"num_streaming_threads", "1"},
    {"memory_reserve_timeout", "100ms"},
    {"allow_overbooking_by_default", "true"},
    {"shuffle_disk", "off"},
    {"shuffle_disk_scratch_dir", "off"},
    {"shuffle_disk_trigger_non_device_bytes", "1GiB"},
    {"shuffle_disk_host_resident_limit", "disabled"},
    {"shuffle_disk_max_pending_write_bytes", "1GiB"},
    {"shuffle_disk_min_spill_chunk_bytes", "0B"},
    {"log", "WARN"},
    {"ucxx_progress_mode", "thread-blocking"},
};

}  // namespace rapidsmpf::config
