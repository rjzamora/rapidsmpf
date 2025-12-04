/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 *
 * TPC-H Query 18 - Pre-filter Optimization
 *
 * This benchmark implements Q18 with a two-phase approach:
 *
 * Phase 1 (blocking): Compute qualifying orderkeys
 *   - Read lineitem -> groupby(l_orderkey, sum(l_quantity)) -> filter(sum > 300)
 *   - All-gather across ranks -> final groupby+filter
 *   - Result: ~171K qualifying orderkeys at SF3000 (tiny!)
 *
 * Phase 2 (streaming): Pre-filter and join
 *   - Read lineitem -> semi-join filter -> all-gather (~684K rows)
 *   - Read orders -> semi-join filter -> all-gather (~171K rows)
 *   - Local join (no shuffle needed - data is tiny!)
 *   - Join with customer -> groupby -> sort -> write
 *
 * Benefits:
 *   - No shuffle needed for lineitem/orders (99.98% data reduction!)
 *   - No fanout node complexity
 *   - Simple memory management
 */

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <optional>

#include <cuda_runtime_api.h>
#include <getopt.h>
#include <mpi.h>

#include <cudf/aggregation.hpp>
#include <cudf/binaryop.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/groupby.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/types.hpp>
#include <cudf/join/filtered_join.hpp>
#include <cudf/join/hash_join.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/sorting.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <rmm/mr/cuda_async_memory_resource.hpp>
#include <rmm/mr/per_device_resource.hpp>
#include <rmm/resource_ref.hpp>

#include <rapidsmpf/communicator/communicator.hpp>
#include <rapidsmpf/communicator/mpi.hpp>
#include <rapidsmpf/communicator/ucxx.hpp>
#include <rapidsmpf/communicator/ucxx_utils.hpp>
#include <rapidsmpf/config.hpp>
#include <rapidsmpf/integrations/cudf/partition.hpp>
#include <rapidsmpf/memory/buffer.hpp>
#include <rapidsmpf/memory/packed_data.hpp>
#include <rapidsmpf/nvtx.hpp>
#include <rapidsmpf/statistics.hpp>
#include <rapidsmpf/streaming/coll/allgather.hpp>
#include <rapidsmpf/streaming/core/channel.hpp>
#include <rapidsmpf/streaming/core/context.hpp>
#include <rapidsmpf/streaming/core/fanout.hpp>
#include <rapidsmpf/streaming/core/node.hpp>
#include <rapidsmpf/streaming/cudf/parquet.hpp>
#include <rapidsmpf/streaming/cudf/table_chunk.hpp>

#include "concatenate.hpp"
#include "join.hpp"
#include "utils.hpp"

namespace {

// ============================================================================
// Table Readers
// ============================================================================

rapidsmpf::streaming::Node read_lineitem(
    std::shared_ptr<rapidsmpf::streaming::Context> ctx,
    std::shared_ptr<rapidsmpf::streaming::Channel> ch_out,
    std::size_t num_producers,
    cudf::size_type num_rows_per_chunk,
    std::string const& input_directory
) {
    auto files = rapidsmpf::ndsh::detail::list_parquet_files(
        rapidsmpf::ndsh::detail::get_table_path(input_directory, "lineitem")
    );
    auto options = cudf::io::parquet_reader_options::builder(cudf::io::source_info(files))
                       .columns({"l_orderkey", "l_quantity"})
                       .build();
    return rapidsmpf::streaming::node::read_parquet(
        ctx, ch_out, num_producers, options, num_rows_per_chunk
    );
}

rapidsmpf::streaming::Node read_orders(
    std::shared_ptr<rapidsmpf::streaming::Context> ctx,
    std::shared_ptr<rapidsmpf::streaming::Channel> ch_out,
    std::size_t num_producers,
    cudf::size_type num_rows_per_chunk,
    std::string const& input_directory
) {
    auto files = rapidsmpf::ndsh::detail::list_parquet_files(
        rapidsmpf::ndsh::detail::get_table_path(input_directory, "orders")
    );
    auto options =
        cudf::io::parquet_reader_options::builder(cudf::io::source_info(files))
            .columns({"o_orderkey", "o_custkey", "o_orderdate", "o_totalprice"})
            .build();
    return rapidsmpf::streaming::node::read_parquet(
        ctx, ch_out, num_producers, options, num_rows_per_chunk
    );
}

rapidsmpf::streaming::Node read_customer(
    std::shared_ptr<rapidsmpf::streaming::Context> ctx,
    std::shared_ptr<rapidsmpf::streaming::Channel> ch_out,
    std::size_t num_producers,
    cudf::size_type num_rows_per_chunk,
    std::string const& input_directory
) {
    auto files = rapidsmpf::ndsh::detail::list_parquet_files(
        rapidsmpf::ndsh::detail::get_table_path(input_directory, "customer")
    );
    auto options = cudf::io::parquet_reader_options::builder(cudf::io::source_info(files))
                       .columns({"c_custkey", "c_name"})
                       .build();
    return rapidsmpf::streaming::node::read_parquet(
        ctx, ch_out, num_producers, options, num_rows_per_chunk
    );
}

// ============================================================================
// Phase 1: Compute Qualifying Orderkeys
// ============================================================================

/**
 * @brief Stage 1: Chunk-wise groupby (NO filter yet!)
 *
 * Computes partial aggregates: groupby(l_orderkey, sum(l_quantity))
 * The same orderkey may appear in multiple chunks, so we can't filter here.
 */
rapidsmpf::streaming::Node chunkwise_groupby_lineitem(
    std::shared_ptr<rapidsmpf::streaming::Context> ctx,
    std::shared_ptr<rapidsmpf::streaming::Channel> ch_in,
    std::shared_ptr<rapidsmpf::streaming::Channel> ch_out
) {
    rapidsmpf::streaming::ShutdownAtExit c{ch_in, ch_out};
    auto mr = ctx->br()->device_mr();
    std::uint64_t sequence = 0;
    std::size_t total_input_rows = 0;
    std::size_t total_output_rows = 0;

    ctx->comm()->logger().print("chunkwise_groupby_lineitem: starting");

    while (true) {
        auto msg = co_await ch_in->receive();
        if (msg.empty()) {
            break;
        }
        co_await ctx->executor()->schedule();
        auto chunk = rapidsmpf::ndsh::to_device(
            ctx, msg.release<rapidsmpf::streaming::TableChunk>()
        );
        auto chunk_stream = chunk.stream();
        auto table = chunk.table_view();
        total_input_rows += static_cast<std::size_t>(table.num_rows());

        // Groupby l_orderkey, sum(l_quantity) - NO FILTER
        auto grouper = cudf::groupby::groupby(
            table.select({0}), cudf::null_policy::EXCLUDE, cudf::sorted::NO
        );
        std::vector<std::unique_ptr<cudf::groupby_aggregation>> aggs;
        aggs.push_back(cudf::make_sum_aggregation<cudf::groupby_aggregation>());
        std::vector<cudf::groupby::aggregation_request> requests;
        requests.push_back(
            cudf::groupby::aggregation_request(table.column(1), std::move(aggs))
        );
        auto [keys, results] = grouper.aggregate(requests, chunk_stream, mr);

        auto result_columns = keys->release();
        for (auto&& r : results) {
            std::ranges::move(r.results, std::back_inserter(result_columns));
        }
        auto grouped_table = std::make_unique<cudf::table>(std::move(result_columns));
        total_output_rows += static_cast<std::size_t>(grouped_table->num_rows());

        if (sequence % 5 == 0) {
            ctx->comm()->logger().print(
                "chunkwise_groupby_lineitem: processed ",
                sequence + 1,
                " chunks, ",
                total_input_rows,
                " input rows -> ",
                total_output_rows,
                " partial aggs"
            );
        }

        if (grouped_table->num_rows() > 0) {
            co_await ch_out->send(
                rapidsmpf::streaming::to_message(
                    sequence++,
                    std::make_unique<rapidsmpf::streaming::TableChunk>(
                        std::move(grouped_table), chunk_stream
                    )
                )
            );
        }
    }

    ctx->comm()->logger().print(
        "chunkwise_groupby_lineitem: done. ",
        total_input_rows,
        " rows -> ",
        total_output_rows,
        " partial aggregates"
    );
    co_await ch_out->drain(ctx->executor());
}

/**
 * @brief Stage 3: Final groupby + filter (after concatenate + all-gather)
 *
 * Merges partial aggregates and filters for sum > threshold.
 * Input: concatenated partial aggregates (l_orderkey, partial_sum)
 * Output: qualifying orderkeys (l_orderkey, total_sum where total_sum > threshold)
 */
rapidsmpf::streaming::Node final_groupby_filter_lineitem(
    std::shared_ptr<rapidsmpf::streaming::Context> ctx,
    std::shared_ptr<rapidsmpf::streaming::Channel> ch_in,
    std::shared_ptr<rapidsmpf::streaming::Channel> ch_out,
    double quantity_threshold
) {
    rapidsmpf::streaming::ShutdownAtExit c{ch_in, ch_out};
    auto mr = ctx->br()->device_mr();

    auto msg = co_await ch_in->receive();
    if (msg.empty()) {
        co_await ch_out->drain(ctx->executor());
        co_return;
    }

    auto chunk =
        rapidsmpf::ndsh::to_device(ctx, msg.release<rapidsmpf::streaming::TableChunk>());
    auto chunk_stream = chunk.stream();
    auto table = chunk.table_view();

    ctx->comm()->logger().debug(
        "final_groupby_filter: input has ", table.num_rows(), " partial aggregates"
    );

    // Final groupby to merge partial sums for same orderkey
    auto grouper = cudf::groupby::groupby(
        table.select({0}), cudf::null_policy::EXCLUDE, cudf::sorted::NO
    );
    std::vector<std::unique_ptr<cudf::groupby_aggregation>> aggs;
    aggs.push_back(cudf::make_sum_aggregation<cudf::groupby_aggregation>());
    std::vector<cudf::groupby::aggregation_request> requests;
    requests.push_back(
        cudf::groupby::aggregation_request(table.column(1), std::move(aggs))
    );
    auto [keys, results] = grouper.aggregate(requests, chunk_stream, mr);

    auto result_columns = keys->release();
    for (auto&& r : results) {
        std::ranges::move(r.results, std::back_inserter(result_columns));
    }
    auto merged_table = std::make_unique<cudf::table>(std::move(result_columns));

    ctx->comm()->logger().debug(
        "final_groupby_filter: merged to ", merged_table->num_rows(), " unique orderkeys"
    );

    // NOW filter for sum > threshold
    auto sum_col = merged_table->view().column(1);
    auto threshold_scalar = cudf::make_numeric_scalar(
        cudf::data_type(cudf::type_id::FLOAT64), chunk_stream, mr
    );
    static_cast<cudf::numeric_scalar<double>*>(threshold_scalar.get())
        ->set_value(quantity_threshold, chunk_stream);

    auto mask = cudf::binary_operation(
        sum_col,
        *threshold_scalar,
        cudf::binary_operator::GREATER,
        cudf::data_type(cudf::type_id::BOOL8),
        chunk_stream,
        mr
    );

    auto filtered_table =
        cudf::apply_boolean_mask(merged_table->view(), mask->view(), chunk_stream, mr);

    ctx->comm()->logger().print(
        "final_groupby_filter: ",
        filtered_table->num_rows(),
        " qualifying orderkeys (sum > ",
        quantity_threshold,
        ")"
    );

    if (filtered_table->num_rows() > 0) {
        // Extract just the orderkey column (column 0) - that's all downstream needs
        std::vector<std::unique_ptr<cudf::column>> cols;
        cols.push_back(
            std::make_unique<cudf::column>(
                filtered_table->view().column(0), chunk_stream, mr
            )
        );
        auto orderkeys_only = std::make_unique<cudf::table>(std::move(cols));

        co_await ch_out->send(
            rapidsmpf::streaming::to_message(
                0,
                std::make_unique<rapidsmpf::streaming::TableChunk>(
                    std::move(orderkeys_only), chunk_stream
                )
            )
        );
    }
    co_await ch_out->drain(ctx->executor());
}

/**
 * @brief All-gather node for Phase 1.
 *
 * Collects partial aggregates from all ranks and outputs concatenated result.
 */
rapidsmpf::streaming::Node allgather_partial_aggregates(
    std::shared_ptr<rapidsmpf::streaming::Context> ctx,
    std::shared_ptr<rapidsmpf::streaming::Channel> ch_in,
    std::shared_ptr<rapidsmpf::streaming::Channel> ch_out,
    rapidsmpf::OpID tag
) {
    rapidsmpf::streaming::ShutdownAtExit c{ch_in, ch_out};

    auto msg = co_await ch_in->receive();

    std::unique_ptr<cudf::table> result;
    rmm::cuda_stream_view stream = cudf::get_default_stream();

    if (ctx->comm()->nranks() > 1) {
        rapidsmpf::streaming::AllGather gatherer{ctx, tag};

        if (!msg.empty()) {
            auto chunk = rapidsmpf::ndsh::to_device(
                ctx, msg.release<rapidsmpf::streaming::TableChunk>()
            );
            stream = chunk.stream();
            auto pack = cudf::pack(chunk.table_view(), stream, ctx->br()->device_mr());
            gatherer.insert(
                0,
                {rapidsmpf::PackedData(
                    std::move(pack.metadata),
                    ctx->br()->move(std::move(pack.gpu_data), stream)
                )}
            );
        }
        gatherer.insert_finished();

        auto packed_data =
            co_await gatherer.extract_all(rapidsmpf::streaming::AllGather::Ordered::NO);

        result = rapidsmpf::unpack_and_concat(
            rapidsmpf::unspill_partitions(
                std::move(packed_data), ctx->br(), true, ctx->statistics()
            ),
            stream,
            ctx->br(),
            ctx->statistics()
        );
    } else {
        // Single rank - just pass through
        if (!msg.empty()) {
            auto chunk = rapidsmpf::ndsh::to_device(
                ctx, msg.release<rapidsmpf::streaming::TableChunk>()
            );
            stream = chunk.stream();
            result = std::make_unique<cudf::table>(
                chunk.table_view(), stream, ctx->br()->device_mr()
            );
        }
    }

    ctx->comm()->logger().debug(
        "allgather_partial_aggregates: ", result ? result->num_rows() : 0, " rows"
    );

    if (result && result->num_rows() > 0) {
        co_await ch_out->send(
            rapidsmpf::streaming::to_message(
                0,
                std::make_unique<rapidsmpf::streaming::TableChunk>(
                    std::move(result), stream
                )
            )
        );
    }
    co_await ch_out->drain(ctx->executor());
}

// ============================================================================
// Phase 2: Pre-filter Pipeline
// ============================================================================

/**
 * @brief Pre-filter table by qualifying orderkeys using semi-join.
 *
 * This node first receives the qualifying orderkeys from ch_orderkeys,
 * builds a hash table, then filters all incoming data chunks from ch_data.
 *
 * @param ch_orderkeys Channel providing the qualifying orderkeys (single message
 * expected)
 * @param ch_data Channel providing data chunks to filter
 * @param ch_out Channel for filtered output
 * @param key_column_idx Which column in input chunks to match against orderkeys.
 */
rapidsmpf::streaming::Node prefilter_by_orderkeys(
    std::shared_ptr<rapidsmpf::streaming::Context> ctx,
    std::shared_ptr<rapidsmpf::streaming::Channel> ch_orderkeys,
    std::shared_ptr<rapidsmpf::streaming::Channel> ch_data,
    std::shared_ptr<rapidsmpf::streaming::Channel> ch_out,
    cudf::size_type key_column_idx
) {
    rapidsmpf::streaming::ShutdownAtExit c{ch_orderkeys, ch_data, ch_out};
    std::uint64_t sequence = 0;

    // First, receive the qualifying orderkeys
    auto orderkeys_msg = co_await ch_orderkeys->receive();
    if (orderkeys_msg.empty()) {
        // No qualifying orderkeys - drain data and exit
        while (true) {
            auto msg = co_await ch_data->receive();
            if (msg.empty())
                break;
        }
        co_await ch_out->drain(ctx->executor());
        co_return;
    }

    auto orderkeys_chunk = rapidsmpf::ndsh::to_device(
        ctx, orderkeys_msg.release<rapidsmpf::streaming::TableChunk>()
    );
    ctx->comm()->logger().print(
        "prefilter: received ",
        orderkeys_chunk.table_view().num_rows(),
        " qualifying orderkeys"
    );

    // Build filtered_join for semi-join (orderkeys is the "right"/build side)
    auto joiner = cudf::filtered_join(
        orderkeys_chunk.table_view(),
        cudf::null_equality::UNEQUAL,
        cudf::set_as_build_table::RIGHT,
        cudf::get_default_stream()
    );

    std::size_t total_input_rows = 0;
    std::size_t total_output_rows = 0;

    while (true) {
        auto msg = co_await ch_data->receive();
        if (msg.empty()) {
            break;
        }
        co_await ctx->executor()->schedule();
        auto chunk = rapidsmpf::ndsh::to_device(
            ctx, msg.release<rapidsmpf::streaming::TableChunk>()
        );
        auto chunk_stream = chunk.stream();
        auto table = chunk.table_view();
        total_input_rows += static_cast<std::size_t>(table.num_rows());

        // Semi-join: get indices of matching rows
        auto match = joiner.semi_join(
            table.select({key_column_idx}), chunk_stream, ctx->br()->device_mr()
        );

        // Gather matching rows - convert device_uvector to column_view
        auto indices = cudf::column_view(
            cudf::data_type{cudf::type_id::INT32},
            static_cast<cudf::size_type>(match->size()),
            match->data(),
            nullptr,  // null mask
            0  // null count
        );
        auto filtered = cudf::gather(
            table, indices, cudf::out_of_bounds_policy::DONT_CHECK, chunk_stream
        );

        total_output_rows += static_cast<std::size_t>(filtered->num_rows());

        if (filtered->num_rows() > 0) {
            co_await ch_out->send(
                rapidsmpf::streaming::to_message(
                    sequence++,
                    std::make_unique<rapidsmpf::streaming::TableChunk>(
                        std::move(filtered), chunk_stream
                    )
                )
            );
        }
    }

    ctx->comm()->logger().debug(
        "prefilter: ",
        total_input_rows,
        " -> ",
        total_output_rows,
        " rows (",
        (total_output_rows * 100.0 / std::max(total_input_rows, std::size_t{1})),
        "%)"
    );

    co_await ch_out->drain(ctx->executor());
}

/**
 * @brief All-gather node: collect from ch_in, all-gather across ranks, send to ch_out.
 */
rapidsmpf::streaming::Node allgather_table(
    std::shared_ptr<rapidsmpf::streaming::Context> ctx,
    std::shared_ptr<rapidsmpf::streaming::Channel> ch_in,
    std::shared_ptr<rapidsmpf::streaming::Channel> ch_out,
    rapidsmpf::OpID tag
) {
    rapidsmpf::streaming::ShutdownAtExit c{ch_in, ch_out};

    auto msg = co_await ch_in->receive();
    if (msg.empty()) {
        co_await ch_out->drain(ctx->executor());
        co_return;
    }

    auto chunk =
        rapidsmpf::ndsh::to_device(ctx, msg.release<rapidsmpf::streaming::TableChunk>());
    auto chunk_stream = chunk.stream();
    auto table = chunk.table_view();

    ctx->comm()->logger().debug("allgather_table: local has ", table.num_rows(), " rows");

    std::unique_ptr<cudf::table> result;
    if (ctx->comm()->nranks() > 1) {
        rapidsmpf::streaming::AllGather gatherer{ctx, tag};

        auto pack = cudf::pack(table, chunk_stream, ctx->br()->device_mr());
        gatherer.insert(
            0,
            {rapidsmpf::PackedData(
                std::move(pack.metadata),
                ctx->br()->move(std::move(pack.gpu_data), chunk_stream)
            )}
        );
        gatherer.insert_finished();

        auto packed_data =
            co_await gatherer.extract_all(rapidsmpf::streaming::AllGather::Ordered::NO);

        result = rapidsmpf::unpack_and_concat(
            rapidsmpf::unspill_partitions(
                std::move(packed_data), ctx->br(), true, ctx->statistics()
            ),
            chunk_stream,
            ctx->br(),
            ctx->statistics()
        );
    } else {
        result =
            std::make_unique<cudf::table>(table, chunk_stream, ctx->br()->device_mr());
    }

    ctx->comm()->logger().debug(
        "allgather_table: gathered has ", result->num_rows(), " rows"
    );

    co_await ch_out->send(
        rapidsmpf::streaming::to_message(
            0,
            std::make_unique<rapidsmpf::streaming::TableChunk>(
                std::move(result), chunk_stream
            )
        )
    );
    co_await ch_out->drain(ctx->executor());
}

/**
 * @brief Local inner join (both inputs are small after pre-filtering).
 *
 * Receives one chunk from each input, joins them, outputs result.
 */
rapidsmpf::streaming::Node local_inner_join(
    std::shared_ptr<rapidsmpf::streaming::Context> ctx,
    std::shared_ptr<rapidsmpf::streaming::Channel> ch_left,
    std::shared_ptr<rapidsmpf::streaming::Channel> ch_right,
    std::shared_ptr<rapidsmpf::streaming::Channel> ch_out,
    std::vector<cudf::size_type> left_on,
    std::vector<cudf::size_type> right_on
) {
    rapidsmpf::streaming::ShutdownAtExit c{ch_left, ch_right, ch_out};

    auto left_msg = co_await ch_left->receive();
    auto right_msg = co_await ch_right->receive();

    if (left_msg.empty() || right_msg.empty()) {
        co_await ch_out->drain(ctx->executor());
        co_return;
    }

    auto left_chunk = rapidsmpf::ndsh::to_device(
        ctx, left_msg.release<rapidsmpf::streaming::TableChunk>()
    );
    auto right_chunk = rapidsmpf::ndsh::to_device(
        ctx, right_msg.release<rapidsmpf::streaming::TableChunk>()
    );

    auto stream = left_chunk.stream();
    auto left_table = left_chunk.table_view();
    auto right_table = right_chunk.table_view();

    ctx->comm()->logger().print(
        "local_inner_join: ", left_table.num_rows(), " x ", right_table.num_rows()
    );

    // Build hash table on right (smaller side typically)
    auto hash_table =
        cudf::hash_join(right_table.select(right_on), cudf::null_equality::EQUAL, stream);

    auto [left_indices, right_indices] = hash_table.inner_join(
        left_table.select(left_on), {}, stream, ctx->br()->device_mr()
    );

    // Gather from both sides using device_span for column_view
    cudf::column_view left_col = cudf::device_span<cudf::size_type const>(*left_indices);
    cudf::column_view right_col =
        cudf::device_span<cudf::size_type const>(*right_indices);

    auto left_gathered = cudf::gather(
        left_table, left_col, cudf::out_of_bounds_policy::DONT_CHECK, stream
    );
    auto right_gathered = cudf::gather(
        right_table.select({1}),  // Only l_quantity from lineitem
        right_col,
        cudf::out_of_bounds_policy::DONT_CHECK,
        stream
    );

    // Concatenate columns: all from left + l_quantity from right
    auto result_cols = left_gathered->release();
    auto right_cols = right_gathered->release();
    for (auto&& col : right_cols) {
        result_cols.push_back(std::move(col));
    }

    auto result = std::make_unique<cudf::table>(std::move(result_cols));
    ctx->comm()->logger().print(
        "local_inner_join: result has ", result->num_rows(), " rows"
    );

    co_await ch_out->send(
        rapidsmpf::streaming::to_message(
            0,
            std::make_unique<rapidsmpf::streaming::TableChunk>(std::move(result), stream)
        )
    );
    co_await ch_out->drain(ctx->executor());
}

// ============================================================================
// Final Processing (reused from q18.cpp)
// ============================================================================

rapidsmpf::streaming::Node reorder_columns(
    std::shared_ptr<rapidsmpf::streaming::Context> ctx,
    std::shared_ptr<rapidsmpf::streaming::Channel> ch_in,
    std::shared_ptr<rapidsmpf::streaming::Channel> ch_out
) {
    rapidsmpf::streaming::ShutdownAtExit c{ch_in, ch_out};
    std::uint64_t seq = 0;
    while (true) {
        auto msg = co_await ch_in->receive();
        if (msg.empty()) {
            break;
        }
        co_await ctx->executor()->schedule();
        auto chunk = rapidsmpf::ndsh::to_device(
            ctx, msg.release<rapidsmpf::streaming::TableChunk>()
        );
        auto table = chunk.table_view();
        auto reordered_table = std::make_unique<cudf::table>(
            table.select({1, 0, 2, 3, 4, 5}), chunk.stream(), ctx->br()->device_mr()
        );
        co_await ch_out->send(
            rapidsmpf::streaming::to_message(
                seq++,
                std::make_unique<rapidsmpf::streaming::TableChunk>(
                    std::move(reordered_table), chunk.stream()
                )
            )
        );
    }
    co_await ch_out->drain(ctx->executor());
}

/**
 * @brief Spillable buffer node.
 *
 * Accumulates all incoming messages in a spillable container.
 * Once input is exhausted, forwards all messages to output.
 * The spill manager can spill buffered messages to host if needed.
 */
rapidsmpf::streaming::Node spillable_buffer(
    std::shared_ptr<rapidsmpf::streaming::Context> ctx,
    std::shared_ptr<rapidsmpf::streaming::Channel> ch_in,
    std::shared_ptr<rapidsmpf::streaming::Channel> ch_out
) {
    rapidsmpf::streaming::ShutdownAtExit c{ch_in, ch_out};
    ctx->comm()->logger().print("spillable_buffer: starting");

    auto spillable = ctx->spillable_messages();
    std::vector<rapidsmpf::streaming::SpillableMessages::MessageId> message_ids;

    while (true) {
        auto msg = co_await ch_in->receive();
        if (msg.empty()) {
            break;
        }
        co_await ctx->executor()->schedule();
        auto id = spillable->insert(std::move(msg));
        message_ids.push_back(id);

        if (message_ids.size() % 10 == 0) {
            ctx->comm()->logger().print(
                "spillable_buffer: buffered ", message_ids.size(), " messages"
            );
        }
    }

    ctx->comm()->logger().print(
        "spillable_buffer: forwarding ", message_ids.size(), " messages"
    );

    for (auto id : message_ids) {
        auto msg = spillable->extract(id);
        co_await ch_out->send(std::move(msg));
    }

    co_await ch_out->drain(ctx->executor());
}

rapidsmpf::streaming::Node chunkwise_groupby_agg(
    std::shared_ptr<rapidsmpf::streaming::Context> ctx,
    std::shared_ptr<rapidsmpf::streaming::Channel> ch_in,
    std::shared_ptr<rapidsmpf::streaming::Channel> ch_out
) {
    rapidsmpf::streaming::ShutdownAtExit c{ch_in, ch_out};
    std::uint64_t sequence = 0;
    while (true) {
        auto msg = co_await ch_in->receive();
        if (msg.empty()) {
            break;
        }
        co_await ctx->executor()->schedule();
        auto chunk = rapidsmpf::ndsh::to_device(
            ctx, msg.release<rapidsmpf::streaming::TableChunk>()
        );
        auto chunk_stream = chunk.stream();
        auto table = chunk.table_view();

        auto grouper = cudf::groupby::groupby(
            table.select({0, 1, 2, 3, 4}), cudf::null_policy::EXCLUDE, cudf::sorted::NO
        );
        std::vector<std::unique_ptr<cudf::groupby_aggregation>> aggs;
        aggs.push_back(cudf::make_sum_aggregation<cudf::groupby_aggregation>());
        std::vector<cudf::groupby::aggregation_request> requests;
        requests.push_back(
            cudf::groupby::aggregation_request(table.column(5), std::move(aggs))
        );
        auto [keys, results] =
            grouper.aggregate(requests, chunk_stream, ctx->br()->device_mr());

        auto result_columns = keys->release();
        for (auto&& r : results) {
            std::ranges::move(r.results, std::back_inserter(result_columns));
        }

        co_await ch_out->send(
            rapidsmpf::streaming::to_message(
                sequence++,
                std::make_unique<rapidsmpf::streaming::TableChunk>(
                    std::make_unique<cudf::table>(std::move(result_columns)), chunk_stream
                )
            )
        );
    }
    co_await ch_out->drain(ctx->executor());
}

rapidsmpf::streaming::Node final_groupby_and_sort(
    std::shared_ptr<rapidsmpf::streaming::Context> ctx,
    std::shared_ptr<rapidsmpf::streaming::Channel> ch_in,
    std::shared_ptr<rapidsmpf::streaming::Channel> ch_out,
    rapidsmpf::OpID allgather_tag
) {
    rapidsmpf::streaming::ShutdownAtExit c{ch_in, ch_out};

    auto msg = co_await ch_in->receive();
    if (msg.empty()) {
        co_await ch_out->drain(ctx->executor());
        co_return;
    }

    auto chunk =
        rapidsmpf::ndsh::to_device(ctx, msg.release<rapidsmpf::streaming::TableChunk>());
    auto stream = chunk.stream();
    auto table = chunk.table_view();

    // Local groupby
    std::unique_ptr<cudf::table> local_result;
    if (table.num_rows() > 0) {
        auto grouper = cudf::groupby::groupby(
            table.select({0, 1, 2, 3, 4}), cudf::null_policy::EXCLUDE, cudf::sorted::NO
        );
        std::vector<std::unique_ptr<cudf::groupby_aggregation>> aggs;
        aggs.push_back(cudf::make_sum_aggregation<cudf::groupby_aggregation>());
        std::vector<cudf::groupby::aggregation_request> requests;
        requests.push_back(
            cudf::groupby::aggregation_request(table.column(5), std::move(aggs))
        );
        auto [keys, results] =
            grouper.aggregate(requests, stream, ctx->br()->device_mr());
        auto cols = keys->release();
        for (auto&& r : results) {
            std::ranges::move(r.results, std::back_inserter(cols));
        }
        local_result = std::make_unique<cudf::table>(std::move(cols));
    }

    // All-gather if multi-rank
    std::unique_ptr<cudf::table> global_result;
    if (ctx->comm()->nranks() > 1 && local_result) {
        rapidsmpf::streaming::AllGather gatherer{ctx, allgather_tag};
        auto pack = cudf::pack(local_result->view(), stream, ctx->br()->device_mr());
        gatherer.insert(
            0,
            {rapidsmpf::PackedData(
                std::move(pack.metadata),
                ctx->br()->move(std::move(pack.gpu_data), stream)
            )}
        );
        gatherer.insert_finished();

        auto packed_data =
            co_await gatherer.extract_all(rapidsmpf::streaming::AllGather::Ordered::NO);

        if (ctx->comm()->rank() == 0) {
            auto gathered = rapidsmpf::unpack_and_concat(
                rapidsmpf::unspill_partitions(
                    std::move(packed_data), ctx->br(), true, ctx->statistics()
                ),
                stream,
                ctx->br(),
                ctx->statistics()
            );

            // Final groupby
            auto grouper = cudf::groupby::groupby(
                gathered->view().select({0, 1, 2, 3, 4}),
                cudf::null_policy::EXCLUDE,
                cudf::sorted::NO
            );
            std::vector<std::unique_ptr<cudf::groupby_aggregation>> aggs;
            aggs.push_back(cudf::make_sum_aggregation<cudf::groupby_aggregation>());
            std::vector<cudf::groupby::aggregation_request> requests;
            requests.push_back(
                cudf::groupby::aggregation_request(
                    gathered->view().column(5), std::move(aggs)
                )
            );
            auto [keys, results] =
                grouper.aggregate(requests, stream, ctx->br()->device_mr());
            auto cols = keys->release();
            for (auto&& r : results) {
                std::ranges::move(r.results, std::back_inserter(cols));
            }
            global_result = std::make_unique<cudf::table>(std::move(cols));
        }
    } else {
        global_result = std::move(local_result);
    }

    // Sort and limit (rank 0 only in multi-rank)
    if (global_result && (ctx->comm()->nranks() == 1 || ctx->comm()->rank() == 0)) {
        auto sorted = cudf::sort_by_key(
            global_result->view(),
            global_result->view().select({4, 3}),  // o_totalprice DESC, o_orderdate ASC
            {cudf::order::DESCENDING, cudf::order::ASCENDING},
            {cudf::null_order::AFTER, cudf::null_order::AFTER},
            stream,
            ctx->br()->device_mr()
        );
        cudf::size_type limit = std::min(100, sorted->num_rows());
        auto limited = cudf::slice(sorted->view(), {0, limit})[0];

        co_await ch_out->send(
            rapidsmpf::streaming::to_message(
                0,
                std::make_unique<rapidsmpf::streaming::TableChunk>(
                    std::make_unique<cudf::table>(
                        limited, stream, ctx->br()->device_mr()
                    ),
                    stream
                )
            )
        );
    }

    co_await ch_out->drain(ctx->executor());
}

rapidsmpf::streaming::Node write_parquet(
    std::shared_ptr<rapidsmpf::streaming::Context> ctx,
    std::shared_ptr<rapidsmpf::streaming::Channel> ch_in,
    std::string output_path
) {
    rapidsmpf::streaming::ShutdownAtExit c{ch_in};
    auto msg = co_await ch_in->receive();
    if (msg.empty()) {
        co_return;
    }
    auto chunk =
        rapidsmpf::ndsh::to_device(ctx, msg.release<rapidsmpf::streaming::TableChunk>());
    auto sink = cudf::io::sink_info(output_path);
    auto builder = cudf::io::parquet_writer_options::builder(sink, chunk.table_view());
    auto metadata = cudf::io::table_input_metadata(chunk.table_view());
    metadata.column_metadata[0].set_name("c_name");
    metadata.column_metadata[1].set_name("c_custkey");
    metadata.column_metadata[2].set_name("o_orderkey");
    metadata.column_metadata[3].set_name("o_orderdate");
    metadata.column_metadata[4].set_name("o_totalprice");
    metadata.column_metadata[5].set_name("col6");
    builder = builder.metadata(metadata);
    cudf::io::write_parquet(builder.build(), chunk.stream());
    ctx->comm()->logger().print(
        "Wrote ", chunk.table_view().num_rows(), " rows to ", output_path
    );
}

}  // namespace

// ============================================================================
// Command Line Parsing
// ============================================================================

struct ProgramOptions {
    int num_streaming_threads{4};  // Increased default for better parallelism
    cudf::size_type num_rows_per_chunk{100'000'000};
    std::uint32_t num_partitions{64};  // Number of shuffle partitions
    std::optional<double> spill_device_limit{std::nullopt};
    std::string output_file;
    std::string input_directory;
};

ProgramOptions parse_options(int argc, char** argv) {
    ProgramOptions options;

    auto print_usage = [&argv]() {
        std::cerr
            << "Usage: " << argv[0] << " [options]\n"
            << "Options:\n"
            << "  --num-streaming-threads <n>  Number of streaming threads (default: 4)\n"
            << "  --num-rows-per-chunk <n>     Number of rows per chunk (default: "
               "100000000)\n"
            << "  --num-partitions <n>         Number of shuffle partitions (default: "
               "64)\n"
            << "  --spill-device-limit <n>     Fractional spill device limit (default: "
               "None)\n"
            << "  --output-file <path>         Output file path (required)\n"
            << "  --input-directory <path>     Input directory path (required)\n"
            << "  --help                       Show this help message\n";
    };

    static struct option long_options[] = {
        {"num-streaming-threads", required_argument, nullptr, 1},
        {"num-rows-per-chunk", required_argument, nullptr, 2},
        {"output-file", required_argument, nullptr, 3},
        {"input-directory", required_argument, nullptr, 4},
        {"help", no_argument, nullptr, 5},
        {"spill-device-limit", required_argument, nullptr, 6},
        {"num-partitions", required_argument, nullptr, 7},
        {nullptr, 0, nullptr, 0}
    };

    int opt;
    int option_index = 0;
    bool saw_output_file = false;
    bool saw_input_directory = false;

    while ((opt = getopt_long(argc, argv, "", long_options, &option_index)) != -1) {
        switch (opt) {
        case 1:
            options.num_streaming_threads = std::atoi(optarg);
            break;
        case 2:
            options.num_rows_per_chunk = std::atoi(optarg);
            break;
        case 3:
            options.output_file = optarg;
            saw_output_file = true;
            break;
        case 4:
            options.input_directory = optarg;
            saw_input_directory = true;
            break;
        case 5:
            print_usage();
            std::exit(0);
        case 6:
            options.spill_device_limit = std::stod(optarg);
            break;
        case 7:
            options.num_partitions = static_cast<std::uint32_t>(std::atoi(optarg));
            break;
        default:
            print_usage();
            std::exit(1);
        }
    }

    if (!saw_output_file || !saw_input_directory) {
        if (!saw_output_file)
            std::cerr << "Error: --output-file is required\n";
        if (!saw_input_directory)
            std::cerr << "Error: --input-directory is required\n";
        print_usage();
        std::exit(1);
    }

    return options;
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    cudaFree(nullptr);
    rapidsmpf::mpi::init(&argc, &argv);
    MPI_Comm mpi_comm;
    RAPIDSMPF_MPI(MPI_Comm_dup(MPI_COMM_WORLD, &mpi_comm));

    auto cmd_options = parse_options(argc, argv);

    auto limit_size = rmm::percent_of_free_device_memory(
        static_cast<std::size_t>(cmd_options.spill_device_limit.value_or(1) * 100)
    );
    rmm::mr::cuda_async_memory_resource mr{};
    auto stats_mr = rapidsmpf::RmmResourceAdaptor(&mr);
    rmm::device_async_resource_ref mr_ref(stats_mr);
    rmm::mr::set_current_device_resource(&stats_mr);
    rmm::mr::set_current_device_resource_ref(mr_ref);

    std::unordered_map<rapidsmpf::MemoryType, rapidsmpf::BufferResource::MemoryAvailable>
        memory_available{};
    if (cmd_options.spill_device_limit.has_value()) {
        memory_available[rapidsmpf::MemoryType::DEVICE] = rapidsmpf::LimitAvailableMemory{
            &stats_mr, static_cast<std::int64_t>(limit_size)
        };
    }

    auto br = std::make_shared<rapidsmpf::BufferResource>(
        stats_mr, std::move(memory_available)
    );
    auto envvars = rapidsmpf::config::get_environment_variables();
    envvars["num_streaming_threads"] = std::to_string(cmd_options.num_streaming_threads);
    auto options = rapidsmpf::config::Options(envvars);
    auto stats = std::make_shared<rapidsmpf::Statistics>(&stats_mr);

    {
        auto comm = rapidsmpf::ucxx::init_using_mpi(mpi_comm, options);
        auto progress =
            std::make_shared<rapidsmpf::ProgressThread>(comm->logger(), stats);
        auto ctx =
            std::make_shared<rapidsmpf::streaming::Context>(options, comm, br, stats);

        comm->logger().print("Q18 Pre-filter Benchmark");
        comm->logger().print(
            "Executor has ", ctx->executor()->thread_count(), " threads"
        );
        comm->logger().print("Executor has ", ctx->comm()->nranks(), " ranks");

        std::string output_path = cmd_options.output_file;

        for (int iteration = 0; iteration < 2; iteration++) {
            auto start = std::chrono::steady_clock::now();

            // ================================================================
            // UNIFIED PIPELINE (Single Lineitem Read)
            // ================================================================
            std::vector<rapidsmpf::streaming::Node> nodes;

            // ----------------------------------------------------------------
            // Read lineitem ONCE and fanout to groupby + spillable_buffer
            // ----------------------------------------------------------------
            auto lineitem = ctx->create_channel();
            nodes.push_back(read_lineitem(
                ctx,
                lineitem,
                4,  // More producers for better parallelism
                cmd_options.num_rows_per_chunk,
                cmd_options.input_directory
            ));

            // Fanout lineitem to: (1) groupby path, (2) spillable buffer for prefilter
            auto lineitem_for_groupby = ctx->create_channel();
            auto lineitem_for_buffer = ctx->create_channel();
            nodes.push_back(
                rapidsmpf::streaming::node::fanout(
                    ctx,
                    lineitem,
                    {lineitem_for_groupby, lineitem_for_buffer},
                    rapidsmpf::streaming::node::FanoutPolicy::UNBOUNDED
                )
            );

            // Spillable buffer holds lineitem data while Phase 1 runs
            auto lineitem_buffered = ctx->create_channel();
            nodes.push_back(
                spillable_buffer(ctx, lineitem_for_buffer, lineitem_buffered)
            );

            // ----------------------------------------------------------------
            // Phase 1: Compute qualifying orderkeys
            // groupby -> SHUFFLE(by orderkey) -> final_groupby_filter -> allgather(tiny)
            // ----------------------------------------------------------------
            std::uint32_t num_partitions = cmd_options.num_partitions;

            auto partial_aggs = ctx->create_channel();
            nodes.push_back(
                chunkwise_groupby_lineitem(ctx, lineitem_for_groupby, partial_aggs)
            );

            // SHUFFLE partial aggregates by orderkey (distributes work!)
            auto partial_aggs_shuffled = ctx->create_channel();
            nodes.push_back(
                rapidsmpf::ndsh::shuffle(
                    ctx,
                    partial_aggs,
                    partial_aggs_shuffled,
                    {0},  // l_orderkey
                    num_partitions,
                    rapidsmpf::OpID{static_cast<rapidsmpf::OpID>(100 + iteration)}
                )
            );

            // Each partition: concat -> groupby -> filter
            auto partial_aggs_concat = ctx->create_channel();
            nodes.push_back(
                rapidsmpf::ndsh::concatenate(
                    ctx,
                    partial_aggs_shuffled,
                    partial_aggs_concat,
                    rapidsmpf::ndsh::ConcatOrder::DONT_CARE
                )
            );

            // Final groupby + filter per partition (parallel across ranks!)
            auto qualifying_orderkeys_local = ctx->create_channel();
            nodes.push_back(final_groupby_filter_lineitem(
                ctx, partial_aggs_concat, qualifying_orderkeys_local, 300.0
            ));

            // All-gather ONLY the tiny result (~5.7K orderkeys at SF300)
            auto qualifying_orderkeys = ctx->create_channel();
            nodes.push_back(allgather_partial_aggregates(
                ctx,
                qualifying_orderkeys_local,
                qualifying_orderkeys,
                rapidsmpf::OpID{static_cast<rapidsmpf::OpID>(101 + iteration)}
            ));

            // ----------------------------------------------------------------
            // Fanout qualifying orderkeys to prefilter nodes
            // ----------------------------------------------------------------
            auto orderkeys_for_lineitem = ctx->create_channel();
            auto orderkeys_for_orders = ctx->create_channel();
            nodes.push_back(
                rapidsmpf::streaming::node::fanout(
                    ctx,
                    qualifying_orderkeys,
                    {orderkeys_for_lineitem, orderkeys_for_orders},
                    rapidsmpf::streaming::node::FanoutPolicy::BOUNDED
                )
            );

            // ----------------------------------------------------------------
            // Phase 2: Pre-filter lineitem (from buffer) and orders, then join
            // ----------------------------------------------------------------

            // Pre-filter lineitem from spillable buffer
            auto lineitem_filtered = ctx->create_channel();
            nodes.push_back(prefilter_by_orderkeys(
                ctx, orderkeys_for_lineitem, lineitem_buffered, lineitem_filtered, 0
            ));

            auto lineitem_concat = ctx->create_channel();
            nodes.push_back(
                rapidsmpf::ndsh::concatenate(
                    ctx,
                    lineitem_filtered,
                    lineitem_concat,
                    rapidsmpf::ndsh::ConcatOrder::DONT_CARE
                )
            );

            auto lineitem_gathered = ctx->create_channel();
            nodes.push_back(allgather_table(
                ctx,
                lineitem_concat,
                lineitem_gathered,
                rapidsmpf::OpID{static_cast<rapidsmpf::OpID>(200 + iteration)}
            ));

            // Read orders and pre-filter
            auto orders_raw = ctx->create_channel();
            nodes.push_back(read_orders(
                ctx,
                orders_raw,
                1,
                cmd_options.num_rows_per_chunk,
                cmd_options.input_directory
            ));

            auto orders_filtered = ctx->create_channel();
            nodes.push_back(prefilter_by_orderkeys(
                ctx, orderkeys_for_orders, orders_raw, orders_filtered, 0
            ));

            auto orders_concat = ctx->create_channel();
            nodes.push_back(
                rapidsmpf::ndsh::concatenate(
                    ctx,
                    orders_filtered,
                    orders_concat,
                    rapidsmpf::ndsh::ConcatOrder::DONT_CARE
                )
            );

            auto orders_gathered = ctx->create_channel();
            nodes.push_back(allgather_table(
                ctx,
                orders_concat,
                orders_gathered,
                rapidsmpf::OpID{static_cast<rapidsmpf::OpID>(201 + iteration)}
            ));

            // Local join: orders x lineitem (both small after pre-filtering)
            auto orders_x_lineitem = ctx->create_channel();
            nodes.push_back(local_inner_join(
                ctx,
                orders_gathered,
                lineitem_gathered,
                orders_x_lineitem,
                {0},  // o_orderkey
                {0}  // l_orderkey
            ));

            // Join with customer (broadcast - orders_x_lineitem is small)
            auto customer = ctx->create_channel();
            nodes.push_back(read_customer(
                ctx,
                customer,
                1,
                cmd_options.num_rows_per_chunk,
                cmd_options.input_directory
            ));

            auto all_joined = ctx->create_channel();
            nodes.push_back(
                rapidsmpf::ndsh::inner_join_broadcast(
                    ctx,
                    customer,
                    orders_x_lineitem,
                    all_joined,
                    {0},  // c_custkey
                    {1},  // o_custkey
                    rapidsmpf::OpID{static_cast<rapidsmpf::OpID>(202 + iteration)},
                    rapidsmpf::ndsh::KeepKeys::YES
                )
            );

            // Reorder columns
            auto reordered = ctx->create_channel();
            nodes.push_back(reorder_columns(ctx, all_joined, reordered));

            // Groupby aggregation
            auto groupby_output = ctx->create_channel();
            nodes.push_back(chunkwise_groupby_agg(ctx, reordered, groupby_output));

            auto concat_groupby = ctx->create_channel();
            nodes.push_back(
                rapidsmpf::ndsh::concatenate(
                    ctx,
                    groupby_output,
                    concat_groupby,
                    rapidsmpf::ndsh::ConcatOrder::DONT_CARE
                )
            );

            // Final groupby, all-gather, sort, limit
            auto final_output = ctx->create_channel();
            nodes.push_back(final_groupby_and_sort(
                ctx,
                concat_groupby,
                final_output,
                rapidsmpf::OpID{static_cast<rapidsmpf::OpID>(203 + iteration)}
            ));

            // Write output
            nodes.push_back(write_parquet(ctx, final_output, output_path));

            // Run pipeline
            auto pipeline_end = std::chrono::steady_clock::now();
            rapidsmpf::streaming::run_streaming_pipeline(std::move(nodes));
            auto compute_end = std::chrono::steady_clock::now();

            std::chrono::duration<double> pipeline_time = pipeline_end - start;
            std::chrono::duration<double> compute_time = compute_end - pipeline_end;

            comm->logger().print(
                "Iteration ",
                iteration,
                " pipeline construction [s]: ",
                pipeline_time.count()
            );
            comm->logger().print(
                "Iteration ", iteration, " compute time [s]: ", compute_time.count()
            );
            comm->logger().print(stats->report());

            RAPIDSMPF_MPI(MPI_Barrier(mpi_comm));
        }
    }

    RAPIDSMPF_MPI(MPI_Comm_free(&mpi_comm));
    RAPIDSMPF_MPI(MPI_Finalize());
    return 0;
}
