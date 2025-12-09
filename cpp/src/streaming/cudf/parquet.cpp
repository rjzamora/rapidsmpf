/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <ranges>

#include <cudf/ast/expressions.hpp>
#include <cudf/io/datasource.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/parquet_metadata.hpp>
#include <cudf/io/types.hpp>
#include <rmm/cuda_stream_view.hpp>

#include <rapidsmpf/cuda_stream.hpp>
#include <rapidsmpf/streaming/core/channel.hpp>
#include <rapidsmpf/streaming/core/coro_utils.hpp>
#include <rapidsmpf/streaming/core/lineariser.hpp>
#include <rapidsmpf/streaming/core/node.hpp>
#include <rapidsmpf/streaming/cudf/parquet.hpp>
#include <rapidsmpf/streaming/cudf/table_chunk.hpp>

namespace rapidsmpf::streaming::node {
namespace {

/**
 * @brief Read a single chunk from a parquet source.
 *
 * @param ctx The execution context to use.
 * @param stream The stream on which to read the chunk.
 * @param options The parquet reader options describing the data to read.
 * @param sequence_number The ordered chunk id to reconstruct original ordering of the
 * data.
 *
 * @return Message representing the read chunk.
 */
Message read_parquet_chunk(
    std::shared_ptr<Context> ctx,
    rmm::cuda_stream_view stream,
    cudf::io::parquet_reader_options options,
    std::uint64_t sequence_number
) {
    auto result = std::make_unique<TableChunk>(
        cudf::io::read_parquet(options, stream, ctx->br()->device_mr()).tbl, stream
    );
    return to_message(sequence_number, std::move(result));
}

struct ChunkDesc {
    std::uint64_t sequence_number;
    std::int64_t skip_rows;
    std::int64_t num_rows;
    cudf::io::source_info source;
    bool read_all_rows;  // If true, ignore skip_rows/num_rows and read all rows
};

/**
 * @brief Read chunks and send them to an output channel.
 *
 * @param ctx Execution context to use.
 * @param ch_out Channel to send output to.
 * @param options Template reader options.
 * @param chunks List of chunks from the input files to read. Processed in order.
 * @param idx Index of the next chunk to process.
 *
 * @return Coroutine representing the processing of all chunks.
 */
Node produce_chunks(
    std::shared_ptr<Context> ctx,
    std::shared_ptr<BoundedQueue> ch_out,
    std::vector<ChunkDesc>& chunks,
    cudf::io::parquet_reader_options options
) {
    // ShutdownAtExit c{ch_out};
    co_await ctx->executor()->schedule();
    for (auto& chunk : chunks) {
        cudf::io::parquet_reader_options chunk_options{options};
        chunk_options.set_source(chunk.source);
        // Only set skip_rows/num_rows if we're not reading all rows
        // This allows the reader to optimize for complete file reads
        if (!chunk.read_all_rows) {
            chunk_options.set_skip_rows(chunk.skip_rows);
            chunk_options.set_num_rows(chunk.num_rows);
        }
        auto stream = ctx->br()->stream_pool().get_stream();
        auto ticket = co_await ch_out->acquire();
        if (!ticket.has_value()) {
            // Semaphore (and hence output channel) shutdown
            break;
        }
        // Having acquire a ticket, let's move to a new thread.
        co_await ctx->executor()->schedule();
        // TODO: This reads the metadata ntasks times.
        // See https://github.com/rapidsai/cudf/issues/20311
        auto sent = co_await ticket->send(
            read_parquet_chunk(ctx, stream, chunk_options, chunk.sequence_number)
        );
        if (!sent) {
            // Output channel is shutdown, no need for more reads.
            break;
        }
    }
    co_await ch_out->drain(ctx->executor());
}
}  // namespace

Node read_parquet(
    std::shared_ptr<Context> ctx,
    std::shared_ptr<Channel> ch_out,
    std::size_t num_producers,
    cudf::io::parquet_reader_options options,
    cudf::size_type num_rows_per_chunk,
    std::unique_ptr<Filter> filter
) {
    ShutdownAtExit c{ch_out};
    co_await ctx->executor()->schedule();
    auto size = static_cast<std::size_t>(ctx->comm()->nranks());
    auto rank = static_cast<std::size_t>(ctx->comm()->rank());
    auto source = options.get_source();
    RAPIDSMPF_EXPECTS(
        source.type() == cudf::io::io_type::FILEPATH, "Only implemented for file sources"
    );
    // TODO: To handle this we need a prefix scan across all the ranks of the total
    // number of rows that would be read by previous ranks.
    RAPIDSMPF_EXPECTS(
        size == 1 || !options.get_num_rows().has_value(),
        "Reading subset of rows not yet supported in multi-rank execution"
    );
    // TODO: To handle this we need a prefix scan across all the ranks of the total
    // number of rows that would be read by previous ranks.
    RAPIDSMPF_EXPECTS(
        size == 1 || options.get_skip_rows() == 0,
        "Skipping rows not yet supported in multi-rank execution"
    );
    auto files = source.filepaths();
    RAPIDSMPF_EXPECTS(files.size() > 0, "Must have at least one file to read");
    RAPIDSMPF_EXPECTS(
        files.size() < std::numeric_limits<int>::max(), "Trying to read too many files"
    );
    RAPIDSMPF_EXPECTS(
        !options.get_filter().has_value(),
        "Do not set filter on options, use the filter argument"
    );
    if (filter != nullptr) {
        options.set_filter(filter->filter);
        // Let's just join all the possible streams here rather than inducing cross-stream
        // deps in the tasks
        cuda_stream_join(
            std::ranges::transform_view(
                std::ranges::iota_view(
                    std::size_t{0}, ctx->br()->stream_pool().get_pool_size()
                ),
                [&](auto i) { return ctx->br()->stream_pool().get_stream(i); }
            ),
            std::ranges::single_view(filter->stream)
        );
    }
    // TODO: Handle case where multiple ranks are reading from a single file.
    int files_per_rank =
        static_cast<int>(files.size() / size + (rank < (files.size() % size)));
    int file_offset = rank * (files.size() / size) + std::min(rank, files.size() % size);
    auto local_files = std::vector(
        files.begin() + file_offset, files.begin() + file_offset + files_per_rank
    );
    std::uint64_t sequence_number = 0;
    std::vector<std::vector<ChunkDesc>> chunks_per_producer(num_producers);
    auto const num_files = local_files.size();
    
    auto to_skip = options.get_skip_rows();
    auto to_read = options.get_num_rows().value_or(std::numeric_limits<int64_t>::max());
    bool has_row_constraints = (to_skip > 0 || to_read < std::numeric_limits<int64_t>::max());
    
    // Fast path: if no row constraints, just divide files among producers without reading metadata
    if (!has_row_constraints) {
        // Simple file-based chunking: accumulate ~num_rows_per_chunk worth of files per chunk
        // Estimate files per chunk based on first file only (to avoid reading all metadata)
        std::size_t files_per_chunk = 1;
        if (num_files > 1) {
            auto first_file_rows =
                cudf::io::read_parquet_metadata(cudf::io::source_info(local_files[0]))
                    .num_rows();
            if (first_file_rows > 0) {
                files_per_chunk = std::max(
                    static_cast<std::size_t>(1),
                    static_cast<std::size_t>(num_rows_per_chunk / first_file_rows)
                );
            }
        }
        
        for (std::size_t file_offset = 0; file_offset < num_files; file_offset += files_per_chunk) {
            std::vector<std::string> chunk_files;
            auto const nchunk_files = std::min(num_files - file_offset, files_per_chunk);
            std::ranges::copy_n(
                local_files.begin() + static_cast<std::int64_t>(file_offset),
                static_cast<std::int64_t>(nchunk_files),
                std::back_inserter(chunk_files)
            );
            auto source = cudf::io::source_info(chunk_files);
            chunks_per_producer[sequence_number % num_producers].emplace_back(
                ChunkDesc{sequence_number, 0, 0, source, true}  // read_all_rows=true
            );
            sequence_number++;
        }
    } else {
        // Slow path: need to read metadata for all files to handle row constraints
        std::vector<int64_t> file_row_counts;
        file_row_counts.reserve(num_files);
        for (const auto& file : local_files) {
            file_row_counts.push_back(
                cudf::io::read_parquet_metadata(cudf::io::source_info(file)).num_rows()
            );
        }
        
        // Greedy bin-packing: accumulate files until we reach ~num_rows_per_chunk
        std::vector<std::string> chunk_files;
        int64_t chunk_total_rows = 0;
        
        for (std::size_t file_idx = 0; file_idx < num_files; file_idx++) {
            const auto& file = local_files[file_idx];
            auto file_rows = file_row_counts[file_idx];
            
            // If this single file exceeds num_rows_per_chunk, subdivide it
            if (file_rows > num_rows_per_chunk && chunk_files.empty()) {
                // Handle large file by subdividing into multiple chunks
                auto source = cudf::io::source_info(file);
                auto chunk_rows = file_rows - to_skip;
                auto chunk_skip_rows = to_skip;
                to_skip = std::max(0l, -chunk_rows);
                
                while (chunk_rows > 0 && to_read > 0) {
                    auto rows_read = std::min({
                        static_cast<int64_t>(num_rows_per_chunk), chunk_rows, to_read
                    });
                    chunks_per_producer[sequence_number % num_producers].emplace_back(
                        ChunkDesc{sequence_number, chunk_skip_rows, rows_read, source, false}
                    );
                    sequence_number++;
                    to_read = std::max(0l, to_read - rows_read);
                    chunk_skip_rows += rows_read;
                    chunk_rows -= rows_read;
                }
            } else {
                // Add file to current chunk
                chunk_files.push_back(file);
                chunk_total_rows += file_rows;
                
                // Emit chunk if we've accumulated enough rows or this is the last file
                bool is_last_file = (file_idx == num_files - 1);
                bool chunk_full = (chunk_total_rows >= num_rows_per_chunk);
                
                if (is_last_file || chunk_full) {
                    auto source = cudf::io::source_info(chunk_files);
                    auto chunk_rows = chunk_total_rows - to_skip;
                    auto chunk_skip_rows = to_skip;
                    to_skip = std::max(0l, -chunk_rows);
                    
                    while (chunk_rows > 0 && to_read > 0) {
                        auto rows_read = std::min({
                            static_cast<int64_t>(num_rows_per_chunk), chunk_rows, to_read
                        });
                        chunks_per_producer[sequence_number % num_producers].emplace_back(
                            ChunkDesc{sequence_number, chunk_skip_rows, rows_read, source, false}
                        );
                        sequence_number++;
                        to_read = std::max(0l, to_read - rows_read);
                        chunk_skip_rows += rows_read;
                        chunk_rows -= rows_read;
                    }
                    
                    // Reset for next chunk
                    chunk_files.clear();
                    chunk_total_rows = 0;
                }
            }
        }
    }
    if (std::ranges::all_of(chunks_per_producer, [](auto&& v) { return v.empty(); })) {
        if (local_files.size() > 0) {
            // If we're on the hook to read some files, but the skip_rows/num_rows setup
            // meant our slice was empty, send an empty table of correct shape.
            // Anyone with no files will just immediately close their output channel.
            auto empty_opts = options;
            empty_opts.set_source(cudf::io::source_info(local_files[0]));
            empty_opts.set_skip_rows(0);
            empty_opts.set_num_rows(0);
            co_await ctx->executor()->schedule(ch_out->send(read_parquet_chunk(
                ctx, ctx->br()->stream_pool().get_stream(), std::move(empty_opts), 0
            )));
        }
    } else {
        std::vector<Node> read_tasks;
        read_tasks.reserve(1 + num_producers);
        auto lineariser = Lineariser(ctx, ch_out, num_producers);
        auto queues = lineariser.get_queues();
        for (std::size_t i = 0; i < num_producers; i++) {
            read_tasks.push_back(
                produce_chunks(ctx, queues[i], chunks_per_producer[i], options)
            );
        }
        read_tasks.push_back(lineariser.drain());
        coro_results(co_await coro::when_all(std::move(read_tasks)));
    }
    co_await ch_out->drain(ctx->executor());
    if (filter != nullptr) {
        // Let's just join all the possible streams here rather than inducing cross-stream
        // deps in the tasks
        cuda_stream_join(
            std::ranges::single_view(filter->stream),
            std::ranges::transform_view(
                std::ranges::iota_view(
                    std::size_t{0}, ctx->br()->stream_pool().get_pool_size()
                ),
                [&](auto i) { return ctx->br()->stream_pool().get_stream(i); }
            )
        );
    }
}
}  // namespace rapidsmpf::streaming::node
