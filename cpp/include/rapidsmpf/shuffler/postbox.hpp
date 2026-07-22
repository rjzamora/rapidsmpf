/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <filesystem>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include <rapidsmpf/error.hpp>
#include <rapidsmpf/shuffler/chunk.hpp>

namespace rapidsmpf::shuffler::detail {

/**
 * @brief A thread-safe container for managing outgoing (to send) chunks.
 */
class ChunksToSend {
  public:
    ChunksToSend() = default;

    /**
     * @brief Insert a chunk into the container.
     *
     * @param c The chunk to insert.
     */
    void insert(std::unique_ptr<Chunk> c);

    /**
     * @brief Extract ready chunks.
     *
     * @note Ready means no stream-ordered work queued on the chunk's data.
     *
     * @return Vector of chunks ready to send.
     */
    [[nodiscard]] std::vector<Chunk> extract_ready();

    /**
     * @brief @return Whether the container is empty.
     */
    [[nodiscard]] bool empty() const;

    /**
     * @brief @return Returns a description of this instance.
     */
    [[nodiscard]] std::string str() const;

  private:
    mutable std::mutex mutex_{};
    std::vector<std::unique_ptr<Chunk>> chunks_{};
};

/**
 * @brief Overloads the stream insertion operator for the ChunksToSend class.
 *
 * This function allows a description of ChunksToSend to be written to an output stream.
 *
 * @param os The output stream to write to.
 * @param obj The object to write.
 * @return A reference to the modified output stream.
 */
inline std::ostream& operator<<(std::ostream& os, ChunksToSend const& obj) {
    os << obj.str();
    return os;
}

/**
 * @brief A thread-safe container for managing received chunks stratified by partition ID.
 */
class ReceivedChunks {
  public:
    /**
     * @brief Policy controlling when received shuffle chunks are staged on disk.
     */
    enum class DiskMode {
        /**
         * @brief Disable disk-backed staging.
         */
        OFF,
        /**
         * @brief Stage host-resident shuffle chunks after disk mode is activated.
         *
         * Stage host-resident shuffle chunks after disk mode is activated by
         * the configured non-device-byte or host-resident-byte thresholds.
         */
        AUTO,
        /**
         * @brief Actively stage received shuffle chunks on disk.
         *
         * Actively stage received shuffle chunks on disk. Device-resident
         * chunks are first moved to host memory, subject to disk-write
         * backpressure.
         */
        FORCE,
    };

    /**
     * @brief Configuration for optional disk-backed storage of received chunks.
     */
    struct DiskOptions {
        /**
         * @brief Disk staging mode.
         */
        DiskMode mode{DiskMode::OFF};
        /**
         * @brief Directory used to create per-shuffle scratch files.
         */
        std::filesystem::path scratch_dir{};
        /**
         * @brief Non-device bytes required before auto mode activates disk staging.
         */
        std::size_t trigger_non_device_bytes{std::numeric_limits<std::size_t>::max()};
        /**
         * @brief Host-resident byte target used to activate or continue disk staging.
         */
        std::size_t host_resident_limit{std::numeric_limits<std::size_t>::max()};
        /**
         * @brief Maximum bytes allowed in asynchronous disk writes at one time.
         */
        std::size_t max_pending_write_bytes{std::numeric_limits<std::size_t>::max()};
        /**
         * @brief Minimum chunk size eligible for disk staging.
         */
        std::size_t min_spill_chunk_bytes{0};
    };

    /**
     * @brief Construct a new container.
     *
     * @param num_keys_hint The number of keys to reserve space for.
     */
    explicit ReceivedChunks(std::size_t num_keys_hint = 0);

    /**
     * @brief Construct a new container with disk-staging configuration.
     *
     * @param num_keys_hint The number of keys to reserve space for.
     * @param disk_options Disk-staging configuration for received chunks.
     */
    ReceivedChunks(std::size_t num_keys_hint, DiskOptions disk_options);

    ~ReceivedChunks();

    ReceivedChunks(ReceivedChunks const&) = delete;
    ReceivedChunks& operator=(ReceivedChunks const&) = delete;
    ReceivedChunks(ReceivedChunks&&) = delete;
    ReceivedChunks& operator=(ReceivedChunks&&) = delete;

    /**
     * @brief Progress asynchronous disk staging.
     *
     * In force mode, ready device-resident chunks are first moved into host
     * memory so they can be submitted to the disk writer. In auto mode, only
     * chunks that are already host-resident are eligible for disk staging.
     *
     * @param br Buffer resource used for device-to-host movement in force mode.
     * @return Number of newly completed disk writes observed.
     */
    std::size_t progress(BufferResource* br = nullptr);

    /**
     * @brief Whether asynchronous disk staging is idle.
     *
     * When disk mode is enabled, this is false while disk writes are pending or while
     * host-resident chunks are still eligible to be staged.
     *
     * @return True if disk staging has no pending or eligible work.
     */
    [[nodiscard]] bool disk_idle() const;

    /**
     * @brief Whether disk-backed storage has been activated.
     *
     * @return True if disk-backed storage is currently active.
     */
    [[nodiscard]] bool disk_mode_enabled() const;

    /**
     * @brief Whether disk-backed storage was configured for this container.
     *
     * @return True if disk-backed storage was configured.
     */
    [[nodiscard]] bool disk_configured() const;

    /**
     * @brief Return a concise disk-staging status string.
     *
     * @return Human-readable disk-staging status.
     */
    [[nodiscard]] std::string disk_status() const;

    /**
     * @brief Return the cumulative bytes observed in non-device shuffle storage.
     *
     * @return Total bytes observed in host, pinned-host, or disk storage.
     */
    [[nodiscard]] std::size_t cumulative_non_device_bytes() const;

    /**
     * @brief Return bytes currently held by pending disk-write jobs.
     *
     * @return Bytes currently held by asynchronous disk-write jobs.
     */
    [[nodiscard]] std::size_t pending_disk_write_bytes() const;

    /**
     * @brief Return bytes currently staged on disk.
     *
     * @return Bytes currently staged on disk.
     */
    [[nodiscard]] std::size_t disk_bytes() const;

    /**
     * @brief Return bytes currently resident in host or pinned-host memory.
     *
     * @return Bytes currently resident in host or pinned-host memory.
     */
    [[nodiscard]] std::size_t host_resident_bytes() const;

    /**
     * @brief Return bytes currently resident in device memory.
     *
     * @return Bytes currently resident in device memory.
     */
    [[nodiscard]] std::size_t device_resident_bytes() const;

    /**
     * @brief Return cumulative bytes submitted to disk staging.
     *
     * @return Total bytes submitted to disk staging.
     */
    [[nodiscard]] std::size_t cumulative_disk_write_bytes() const;

    /**
     * @brief Return cumulative bytes read back from disk staging.
     *
     * @return Total bytes read back from disk staging.
     */
    [[nodiscard]] std::size_t cumulative_disk_read_bytes() const;

  private:
    class DiskWriter;
    struct DiskRecord;
    struct Entry;

    void reserve_pigeonhole(std::size_t num_keys_hint) {
        if (num_keys_hint > 0) {
            pigeonhole_.reserve(num_keys_hint);
        }
    }

  public:
    /**
     * @brief Insert a chunk.
     *
     * @param chunk The chunk to insert.
     */
    void insert(Chunk&& chunk);

    /**
     * @brief Check whether the specified partition contains any chunks.
     *
     * @param pid Identifier of the partition to query.
     * @return True if the partition contains no chunks, false otherwise.
     *
     * @note The result reflects a snapshot at the time of the call and may change
     * immediately afterward.
     */
    [[nodiscard]] bool is_empty(PartID pid) const;

    /**
     * @brief Extracts all chunks associated with a specific partition.
     *
     * @param pid The ID of the partition.
     * @param br Buffer resource used to read disk-staged chunks back into memory.
     * @return A vector of chunks.
     *
     * @throws std::out_of_range If the partition is not found.
     */
    [[nodiscard]] std::vector<Chunk> extract(PartID pid, BufferResource* br);

    /**
     * @brief Checks if the container is empty.
     *
     * @return `true` if the container is empty, `false` otherwise.
     *
     * @note The result reflects a snapshot at the time of the call and may change
     * immediately afterward.
     */
    [[nodiscard]] bool empty() const;

    /**
     * @brief @return A description of this container.
     */
    [[nodiscard]] std::string str() const;

    /**
     * @brief Spill device data.
     *
     * The spilling is stream ordered by the spilled buffers' CUDA streams.
     *
     * @param br The buffer resource for host and device allocations.
     * @param amount Requested amount of data to spill in bytes.
     * @return Actual amount of data spilled in bytes.
     */
    [[nodiscard]] std::size_t spill(BufferResource* br, std::size_t amount);

  private:
    void note_memory_chunk_added(Chunk const& chunk);
    void note_memory_chunk_removed(Chunk const& chunk);
    void maybe_enable_disk_mode();
    bool is_eligible_for_disk(Entry const& entry) const;
    bool is_force_device_staging_candidate(Entry const& entry) const;
    bool has_disk_staging_work() const;
    bool can_move_device_chunk_to_host(std::size_t size) const;
    std::size_t poll_completed_disk_writes();
    std::size_t stage_device_chunks_for_force(BufferResource* br);
    void submit_disk_writes();

    // TODO: more fine-grained locking e.g. by locking each partition individually.
    mutable std::mutex mutex_;
    DiskOptions disk_options_{};
    bool disk_mode_enabled_{false};
    std::unique_ptr<DiskWriter> disk_writer_{};
    std::size_t device_resident_bytes_{0};
    std::size_t host_resident_bytes_{0};
    std::size_t pending_disk_write_bytes_{0};
    std::size_t disk_bytes_{0};
    std::size_t cumulative_non_device_bytes_{0};
    std::size_t cumulative_disk_write_bytes_{0};
    std::size_t cumulative_disk_read_bytes_{0};
    std::unordered_map<PartID, std::vector<Entry>>
        pigeonhole_;  ///< Storage for chunks, stratified by partition ID.
};

/**
 * @brief Overloads the stream insertion operator for the ReceivedChunks class.
 *
 * This function allows a description of ReceivedChunks be written to an output stream.
 *
 * @param os The output stream to write to.
 * @param obj The object to write.
 * @return A reference to the modified output stream.
 */
inline std::ostream& operator<<(std::ostream& os, ReceivedChunks const& obj) {
    os << obj.str();
    return os;
}

}  // namespace rapidsmpf::shuffler::detail
