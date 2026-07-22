/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <exception>
#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <sstream>
#include <thread>
#include <utility>

#include <unistd.h>

#include <rapidsmpf/communicator/communicator.hpp>
#include <rapidsmpf/error.hpp>
#include <rapidsmpf/memory/buffer.hpp>
#include <rapidsmpf/memory/memory_type.hpp>
#include <rapidsmpf/nvtx.hpp>
#include <rapidsmpf/shuffler/chunk.hpp>
#include <rapidsmpf/shuffler/postbox.hpp>
#include <rapidsmpf/utils/misc.hpp>

namespace rapidsmpf::shuffler::detail {

struct ReceivedChunks::DiskRecord {
    std::uint64_t offset{0};
    std::uint64_t metadata_size{0};
    std::uint64_t data_size{0};
};

class ReceivedChunks::DiskWriter {
  public:
    struct WriteState {
        mutable std::mutex mutex;
        std::condition_variable cv;
        bool done{false};
        DiskRecord record{};
        std::exception_ptr error{};
    };

    explicit DiskWriter(std::filesystem::path scratch_dir)
        : work_dir_{make_work_dir(std::move(scratch_dir))},
          data_file_{work_dir_ / "chunks.bin"},
          out_{data_file_, std::ios::binary | std::ios::out | std::ios::trunc} {
        RAPIDSMPF_EXPECTS(out_.is_open(), "failed to open shuffle disk spill file");
        worker_ = std::thread{[this] { run(); }};
    }

    ~DiskWriter() noexcept {
        shutdown();
        std::error_code ec;
        std::filesystem::remove_all(work_dir_, ec);
    }

    DiskWriter(DiskWriter const&) = delete;
    DiskWriter& operator=(DiskWriter const&) = delete;
    DiskWriter(DiskWriter&&) = delete;
    DiskWriter& operator=(DiskWriter&&) = delete;

    std::shared_ptr<WriteState> submit(Chunk&& chunk) {
        RAPIDSMPF_EXPECTS(chunk.data_size() > 0, "cannot stage an empty chunk on disk");
        RAPIDSMPF_EXPECTS(chunk.is_data_buffer_set(), "chunk has no data buffer");
        RAPIDSMPF_EXPECTS(
            contains(Buffer::host_buffer_types, chunk.data_memory_type()),
            "only host-resident chunks can be staged on disk"
        );
        RAPIDSMPF_EXPECTS(chunk.is_ready(), "chunk data is not ready for disk staging");

        auto state = std::make_shared<WriteState>();
        {
            std::lock_guard lock(mutex_);
            jobs_.push_back(Job{.chunk = std::move(chunk), .state = state});
        }
        cv_.notify_one();
        return state;
    }

    std::optional<DiskRecord> try_get(std::shared_ptr<WriteState> const& state) const {
        std::lock_guard lock(state->mutex);
        if (!state->done) {
            return std::nullopt;
        }
        if (state->error) {
            std::rethrow_exception(state->error);
        }
        return state->record;
    }

    DiskRecord wait(std::shared_ptr<WriteState> const& state) const {
        std::unique_lock lock(state->mutex);
        state->cv.wait(lock, [&] { return state->done; });
        if (state->error) {
            std::rethrow_exception(state->error);
        }
        return state->record;
    }

    Chunk load(DiskRecord const& record, BufferResource* br) const {
        RAPIDSMPF_EXPECTS(br != nullptr, "the buffer resource pointer cannot be NULL");
        std::ifstream in(data_file_, std::ios::binary | std::ios::in);
        RAPIDSMPF_EXPECTS(in.is_open(), "failed to open shuffle disk spill file");

        in.seekg(static_cast<std::streamoff>(record.offset));
        RAPIDSMPF_EXPECTS(!in.fail(), "failed to seek shuffle disk spill file");

        auto metadata =
            std::vector<std::uint8_t>(safe_cast<std::size_t>(record.metadata_size));
        in.read(
            reinterpret_cast<char*>(metadata.data()),
            static_cast<std::streamsize>(metadata.size())
        );
        RAPIDSMPF_EXPECTS(!in.fail(), "failed to read shuffle chunk metadata from disk");

        auto reservation = br->reserve_or_fail(
            safe_cast<std::size_t>(record.data_size), MemoryType::HOST
        );
        auto data = br->make_buffer(
            safe_cast<std::size_t>(record.data_size),
            br->stream_pool()->get_stream(),
            reservation
        );
        if (record.data_size > 0) {
            auto* ptr = data->exclusive_data_access();
            try {
                in.read(
                    reinterpret_cast<char*>(ptr),
                    static_cast<std::streamsize>(record.data_size)
                );
                data->unlock();
            } catch (...) {
                data->unlock();
                throw;
            }
            RAPIDSMPF_EXPECTS(
                !in.fail(), "failed to read shuffle chunk payload from disk"
            );
        }

        return Chunk::deserialize(metadata, br, false, std::move(data));
    }

  private:
    struct Job {
        Chunk chunk;
        std::shared_ptr<WriteState> state;
    };

    static std::filesystem::path make_work_dir(std::filesystem::path scratch_dir) {
        static std::atomic<std::uint64_t> counter{0};
        RAPIDSMPF_EXPECTS(
            !scratch_dir.empty(), "shuffle disk scratch directory is empty"
        );
        auto work_dir =
            scratch_dir
            / ("rapidsmpf-shuffle-" + std::to_string(::getpid()) + "-"
               + std::to_string(counter.fetch_add(1, std::memory_order_relaxed)));
        std::error_code ec;
        std::filesystem::create_directories(work_dir, ec);
        RAPIDSMPF_EXPECTS(
            !ec, "failed to create shuffle disk scratch directory: " + ec.message()
        );
        return work_dir;
    }

    DiskRecord write(Chunk&& chunk) {
        auto const metadata = chunk.serialize();
        auto data = chunk.release_data_buffer();

        auto const pos = out_.tellp();
        RAPIDSMPF_EXPECTS(pos >= std::streampos{0}, "failed to query spill file offset");
        DiskRecord record{
            .offset = static_cast<std::uint64_t>(pos),
            .metadata_size = safe_cast<std::uint64_t>(metadata->size()),
            .data_size = safe_cast<std::uint64_t>(data->size),
        };

        out_.write(
            reinterpret_cast<char const*>(metadata->data()),
            static_cast<std::streamsize>(metadata->size())
        );
        RAPIDSMPF_EXPECTS(out_.good(), "failed to write shuffle chunk metadata to disk");

        if (data->size > 0) {
            auto const* ptr = data->exclusive_data_access();
            try {
                out_.write(
                    reinterpret_cast<char const*>(ptr),
                    static_cast<std::streamsize>(data->size)
                );
                data->unlock();
            } catch (...) {
                data->unlock();
                throw;
            }
            RAPIDSMPF_EXPECTS(
                out_.good(), "failed to write shuffle chunk payload to disk"
            );
        }
        out_.flush();
        RAPIDSMPF_EXPECTS(out_.good(), "failed to flush shuffle chunk to disk");
        return record;
    }

    void run() noexcept {
        while (true) {
            std::optional<Job> job;
            {
                std::unique_lock lock(mutex_);
                cv_.wait(lock, [&] { return stop_ || !jobs_.empty(); });
                if (stop_ && jobs_.empty()) {
                    break;
                }
                job.emplace(std::move(jobs_.front()));
                jobs_.pop_front();
            }

            try {
                DiskRecord record = write(std::move(job->chunk));
                {
                    std::lock_guard lock(job->state->mutex);
                    job->state->record = record;
                    job->state->done = true;
                }
            } catch (...) {
                {
                    std::lock_guard lock(job->state->mutex);
                    job->state->error = std::current_exception();
                    job->state->done = true;
                }
            }
            job->state->cv.notify_all();
        }
    }

    void shutdown() noexcept {
        {
            std::lock_guard lock(mutex_);
            stop_ = true;
        }
        cv_.notify_one();
        if (worker_.joinable()) {
            worker_.join();
        }
        out_.close();
    }

    std::filesystem::path work_dir_;
    std::filesystem::path data_file_;
    std::ofstream out_;
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    std::deque<Job> jobs_;
    bool stop_{false};
    std::thread worker_;
};

struct ReceivedChunks::Entry {
    enum class State {
        MEMORY,
        PENDING_DISK,
        ON_DISK,
    };

    static Entry memory(Chunk&& chunk) {
        Entry ret;
        ret.state = State::MEMORY;
        ret.data_size = chunk.data_size();
        ret.chunk = std::make_unique<Chunk>(std::move(chunk));
        return ret;
    }

    State state{State::MEMORY};
    std::size_t data_size{0};
    std::unique_ptr<Chunk> chunk{};
    std::shared_ptr<DiskWriter::WriteState> pending{};
    DiskRecord record{};
};

void ChunksToSend::insert(std::unique_ptr<Chunk> c) {
    std::lock_guard lock(mutex_);
    chunks_.push_back(std::move(c));
}

std::vector<Chunk> ChunksToSend::extract_ready() {
    std::lock_guard lock(mutex_);
    std::vector<Chunk> result;
    for (auto&& chunk : chunks_) {
        if (!chunk->is_ready()) {
            break;
        }
        auto c = std::move(chunk);
        result.emplace_back(std::move(*c));
    }
    std::erase(chunks_, nullptr);
    return result;
}

bool ChunksToSend::empty() const {
    std::lock_guard lock(mutex_);
    return chunks_.empty();
}

std::string ChunksToSend::str() const {
    std::lock_guard const lock(mutex_);
    std::stringstream ss;
    ss << "ChunksToSend(";
    for (auto const& chunk : chunks_) {
        ss << *chunk << ", ";
    }
    ss << ")";
    return ss.str();
}

ReceivedChunks::ReceivedChunks(std::size_t num_keys_hint)
    : ReceivedChunks(num_keys_hint, DiskOptions{}) {}

ReceivedChunks::ReceivedChunks(std::size_t num_keys_hint, DiskOptions disk_options)
    : disk_options_{std::move(disk_options)} {
    reserve_pigeonhole(num_keys_hint);
    if (disk_options_.mode == DiskMode::FORCE) {
        RAPIDSMPF_EXPECTS(
            !disk_options_.scratch_dir.empty(),
            "shuffle disk mode 'force' requires a scratch directory"
        );
        disk_writer_ = std::make_unique<DiskWriter>(disk_options_.scratch_dir);
        disk_mode_enabled_ = true;
    } else if (disk_options_.mode == DiskMode::AUTO && disk_options_.scratch_dir.empty())
    {
        disk_options_.mode = DiskMode::OFF;
    }
}

ReceivedChunks::~ReceivedChunks() = default;

std::size_t ReceivedChunks::progress(BufferResource* br) {
    std::lock_guard lock(mutex_);
    auto completed = poll_completed_disk_writes();
    maybe_enable_disk_mode();
    stage_device_chunks_for_force(br);
    submit_disk_writes();
    return completed;
}

bool ReceivedChunks::disk_idle() const {
    std::lock_guard lock(mutex_);
    if (!disk_mode_enabled_) {
        return true;
    }
    return pending_disk_write_bytes_ == 0 && !has_disk_staging_work();
}

bool ReceivedChunks::disk_mode_enabled() const {
    std::lock_guard lock(mutex_);
    return disk_mode_enabled_;
}

bool ReceivedChunks::disk_configured() const {
    std::lock_guard lock(mutex_);
    return disk_options_.mode != DiskMode::OFF;
}

std::string ReceivedChunks::disk_status() const {
    std::lock_guard lock(mutex_);
    auto mode = [](DiskMode mode) {
        switch (mode) {
        case DiskMode::OFF:
            return "off";
        case DiskMode::AUTO:
            return "auto";
        case DiskMode::FORCE:
            return "force";
        }
        return "unknown";
    };
    std::stringstream ss;
    ss << "mode=" << mode(disk_options_.mode)
       << ", enabled=" << (disk_mode_enabled_ ? "true" : "false")
       << ", scratch_dir=" << disk_options_.scratch_dir.string()
       << ", device_resident_bytes=" << device_resident_bytes_
       << ", host_resident_bytes=" << host_resident_bytes_
       << ", pending_disk_write_bytes=" << pending_disk_write_bytes_
       << ", disk_bytes=" << disk_bytes_
       << ", cumulative_non_device_bytes=" << cumulative_non_device_bytes_
       << ", cumulative_disk_write_bytes=" << cumulative_disk_write_bytes_
       << ", cumulative_disk_read_bytes=" << cumulative_disk_read_bytes_;
    return ss.str();
}

std::size_t ReceivedChunks::cumulative_non_device_bytes() const {
    std::lock_guard lock(mutex_);
    return cumulative_non_device_bytes_;
}

std::size_t ReceivedChunks::pending_disk_write_bytes() const {
    std::lock_guard lock(mutex_);
    return pending_disk_write_bytes_;
}

std::size_t ReceivedChunks::disk_bytes() const {
    std::lock_guard lock(mutex_);
    return disk_bytes_;
}

std::size_t ReceivedChunks::host_resident_bytes() const {
    std::lock_guard lock(mutex_);
    return host_resident_bytes_;
}

std::size_t ReceivedChunks::device_resident_bytes() const {
    std::lock_guard lock(mutex_);
    return device_resident_bytes_;
}

std::size_t ReceivedChunks::cumulative_disk_write_bytes() const {
    std::lock_guard lock(mutex_);
    return cumulative_disk_write_bytes_;
}

std::size_t ReceivedChunks::cumulative_disk_read_bytes() const {
    std::lock_guard lock(mutex_);
    return cumulative_disk_read_bytes_;
}

void ReceivedChunks::insert(Chunk&& chunk) {
    auto key = chunk.part_id();
    std::lock_guard const lock(mutex_);
    auto entry = Entry::memory(std::move(chunk));
    note_memory_chunk_added(*entry.chunk);
    pigeonhole_[key].emplace_back(std::move(entry));
    poll_completed_disk_writes();
    maybe_enable_disk_mode();
    submit_disk_writes();
}

bool ReceivedChunks::is_empty(PartID pid) const {
    std::lock_guard const lock(mutex_);
    return !pigeonhole_.contains(pid);
}

std::vector<Chunk> ReceivedChunks::extract(PartID pid, BufferResource* br) {
    std::vector<Entry> entries;
    {
        std::lock_guard lock(mutex_);
        poll_completed_disk_writes();
        entries = extract_value(pigeonhole_, pid);
        for (auto const& entry : entries) {
            switch (entry.state) {
            case Entry::State::MEMORY:
                note_memory_chunk_removed(*entry.chunk);
                break;
            case Entry::State::PENDING_DISK:
                // Accounted after the pending write completes below.
                break;
            case Entry::State::ON_DISK:
                RAPIDSMPF_EXPECTS(disk_bytes_ >= entry.data_size, "corrupt disk stats");
                disk_bytes_ -= entry.data_size;
                break;
            }
        }
    }

    std::vector<Chunk> ret;
    ret.reserve(entries.size());
    for (auto& entry : entries) {
        switch (entry.state) {
        case Entry::State::MEMORY:
            ret.emplace_back(std::move(*entry.chunk));
            break;
        case Entry::State::PENDING_DISK:
            {
                auto record = disk_writer_->wait(entry.pending);
                {
                    std::lock_guard lock(mutex_);
                    RAPIDSMPF_EXPECTS(
                        pending_disk_write_bytes_ >= entry.data_size,
                        "corrupt pending disk stats"
                    );
                    pending_disk_write_bytes_ -= entry.data_size;
                    cumulative_disk_read_bytes_ += entry.data_size;
                }
                ret.emplace_back(disk_writer_->load(record, br));
                break;
            }
        case Entry::State::ON_DISK:
            {
                std::lock_guard lock(mutex_);
                cumulative_disk_read_bytes_ += entry.data_size;
            }
            ret.emplace_back(disk_writer_->load(entry.record, br));
            break;
        }
    }
    return ret;
}

bool ReceivedChunks::empty() const {
    std::lock_guard const lock(mutex_);
    return pigeonhole_.empty();
}

std::size_t ReceivedChunks::spill(BufferResource* br, std::size_t amount) {
    RAPIDSMPF_NVTX_FUNC_RANGE(amount);
    std::lock_guard lock(mutex_);
    poll_completed_disk_writes();
    // TODO: use a clever strategy to decided which chunks to spill.
    std::size_t total_spilled{0};
    for (auto& [_, entries] : pigeonhole_) {
        for (auto& entry : entries) {
            if (entry.state != Entry::State::MEMORY) {
                continue;
            }
            auto& chunk = *entry.chunk;
            auto const size = chunk.data_size();
            if (size == 0 || !chunk.is_data_buffer_set()
                || chunk.data_memory_type() != MemoryType::DEVICE)
            {
                continue;
            }
            auto reservation = br->reserve_or_fail(size, SPILL_TARGET_MEMORY_TYPES);
            chunk.set_data_buffer(br->move(chunk.release_data_buffer(), reservation));
            RAPIDSMPF_EXPECTS(device_resident_bytes_ >= size, "corrupt device stats");
            device_resident_bytes_ -= size;
            host_resident_bytes_ += size;
            cumulative_non_device_bytes_ += size;
            if ((total_spilled += size) >= amount) {
                break;
            }
        }
        if (total_spilled >= amount) {
            break;
        }
    }
    maybe_enable_disk_mode();
    submit_disk_writes();
    RAPIDSMPF_NVTX_MARKER("ReceivedChunks::spill::total_spilled", total_spilled);
    return total_spilled;
}

void ReceivedChunks::note_memory_chunk_added(Chunk const& chunk) {
    if (chunk.data_size() == 0 || !chunk.is_data_buffer_set()) {
        return;
    }
    if (chunk.data_memory_type() == MemoryType::DEVICE) {
        device_resident_bytes_ += chunk.data_size();
    } else {
        host_resident_bytes_ += chunk.data_size();
        cumulative_non_device_bytes_ += chunk.data_size();
    }
}

void ReceivedChunks::note_memory_chunk_removed(Chunk const& chunk) {
    if (chunk.data_size() == 0 || !chunk.is_data_buffer_set()) {
        return;
    }
    if (chunk.data_memory_type() == MemoryType::DEVICE) {
        RAPIDSMPF_EXPECTS(
            device_resident_bytes_ >= chunk.data_size(), "corrupt device stats"
        );
        device_resident_bytes_ -= chunk.data_size();
    } else {
        RAPIDSMPF_EXPECTS(
            host_resident_bytes_ >= chunk.data_size(), "corrupt host stats"
        );
        host_resident_bytes_ -= chunk.data_size();
    }
}

void ReceivedChunks::maybe_enable_disk_mode() {
    if (disk_mode_enabled_ || disk_options_.mode == DiskMode::OFF) {
        return;
    }
    bool const should_enable =
        disk_options_.mode == DiskMode::FORCE
        || cumulative_non_device_bytes_ >= disk_options_.trigger_non_device_bytes
        || host_resident_bytes_ >= disk_options_.host_resident_limit;
    if (!should_enable) {
        return;
    }
    RAPIDSMPF_EXPECTS(
        !disk_options_.scratch_dir.empty(),
        "shuffle disk mode requires a scratch directory"
    );
    disk_writer_ = std::make_unique<DiskWriter>(disk_options_.scratch_dir);
    disk_mode_enabled_ = true;
}

bool ReceivedChunks::is_eligible_for_disk(Entry const& entry) const {
    if (!disk_mode_enabled_ || entry.state != Entry::State::MEMORY || !entry.chunk) {
        return false;
    }
    auto const& chunk = *entry.chunk;
    return chunk.data_size() >= disk_options_.min_spill_chunk_bytes
           && chunk.data_size() > 0 && chunk.is_data_buffer_set()
           && chunk.data_memory_type() != MemoryType::DEVICE;
}

bool ReceivedChunks::is_force_device_staging_candidate(Entry const& entry) const {
    if (!disk_mode_enabled_ || disk_options_.mode != DiskMode::FORCE
        || entry.state != Entry::State::MEMORY || !entry.chunk)
    {
        return false;
    }
    auto const& chunk = *entry.chunk;
    return chunk.data_size() >= disk_options_.min_spill_chunk_bytes
           && chunk.data_size() > 0 && chunk.is_data_buffer_set()
           && chunk.data_memory_type() == MemoryType::DEVICE;
}

bool ReceivedChunks::has_disk_staging_work() const {
    for (auto const& [_, entries] : pigeonhole_) {
        for (auto const& entry : entries) {
            if (is_eligible_for_disk(entry) || is_force_device_staging_candidate(entry)) {
                return true;
            }
        }
    }
    return false;
}

bool ReceivedChunks::can_move_device_chunk_to_host(std::size_t size) const {
    if (host_resident_bytes_ == 0 && pending_disk_write_bytes_ == 0) {
        return true;
    }
    auto const max_host_backed_bytes = disk_options_.max_pending_write_bytes;
    if (host_resident_bytes_ >= max_host_backed_bytes) {
        return false;
    }
    auto const remaining_after_host = max_host_backed_bytes - host_resident_bytes_;
    return pending_disk_write_bytes_ < remaining_after_host
           && size <= remaining_after_host - pending_disk_write_bytes_;
}

std::size_t ReceivedChunks::poll_completed_disk_writes() {
    if (!disk_mode_enabled_) {
        return 0;
    }
    std::size_t completed{0};
    for (auto& [_, entries] : pigeonhole_) {
        for (auto& entry : entries) {
            if (entry.state != Entry::State::PENDING_DISK) {
                continue;
            }
            auto record = disk_writer_->try_get(entry.pending);
            if (!record.has_value()) {
                continue;
            }
            RAPIDSMPF_EXPECTS(
                pending_disk_write_bytes_ >= entry.data_size, "corrupt pending disk stats"
            );
            pending_disk_write_bytes_ -= entry.data_size;
            disk_bytes_ += entry.data_size;
            entry.record = *record;
            entry.pending.reset();
            entry.state = Entry::State::ON_DISK;
            ++completed;
        }
    }
    return completed;
}

std::size_t ReceivedChunks::stage_device_chunks_for_force(BufferResource* br) {
    if (!disk_mode_enabled_ || disk_options_.mode != DiskMode::FORCE) {
        return 0;
    }
    if (br == nullptr) {
        return 0;
    }

    std::size_t total_staged{0};
    for (auto& [_, entries] : pigeonhole_) {
        for (auto& entry : entries) {
            if (!is_force_device_staging_candidate(entry) || !entry.chunk->is_ready()) {
                continue;
            }
            auto& chunk = *entry.chunk;
            auto const size = chunk.data_size();
            if (!can_move_device_chunk_to_host(size)) {
                continue;
            }
            auto reservation = br->reserve_or_fail(size, SPILL_TARGET_MEMORY_TYPES);
            chunk.set_data_buffer(br->move(chunk.release_data_buffer(), reservation));
            RAPIDSMPF_EXPECTS(device_resident_bytes_ >= size, "corrupt device stats");
            device_resident_bytes_ -= size;
            host_resident_bytes_ += size;
            cumulative_non_device_bytes_ += size;
            total_staged += size;
        }
    }
    return total_staged;
}

void ReceivedChunks::submit_disk_writes() {
    if (!disk_mode_enabled_) {
        return;
    }
    for (auto& [_, entries] : pigeonhole_) {
        for (auto& entry : entries) {
            if (!is_eligible_for_disk(entry) || !entry.chunk->is_ready()) {
                continue;
            }
            auto const size = entry.chunk->data_size();
            if (pending_disk_write_bytes_ > 0
                && pending_disk_write_bytes_ + size
                       > disk_options_.max_pending_write_bytes)
            {
                continue;
            }
            RAPIDSMPF_EXPECTS(host_resident_bytes_ >= size, "corrupt host stats");
            host_resident_bytes_ -= size;
            pending_disk_write_bytes_ += size;
            cumulative_disk_write_bytes_ += size;
            entry.pending = disk_writer_->submit(std::move(*entry.chunk));
            entry.chunk.reset();
            entry.state = Entry::State::PENDING_DISK;
        }
    }
}

std::string ReceivedChunks::str() const {
    if (empty()) {
        return "ReceivedChunks()";
    }
    std::lock_guard const lock(mutex_);
    std::stringstream ss;
    ss << "ReceivedChunks(";
    for (auto const& [key, entries] : pigeonhole_) {
        ss << "k=" << key << ": [";
        for (auto const& entry : entries) {
            switch (entry.state) {
            case Entry::State::MEMORY:
                ss << *entry.chunk;
                break;
            case Entry::State::PENDING_DISK:
                ss << "PendingDisk(data_size=" << entry.data_size << ")";
                break;
            case Entry::State::ON_DISK:
                ss << "OnDisk(data_size=" << entry.data_size << ")";
                break;
            }
            ss << ", ";
        }
        ss << "\b\b], ";
    }
    ss << "\b\b)";
    return ss.str();
}

}  // namespace rapidsmpf::shuffler::detail
