# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Join integration for Dask Distributed clusters."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Any, Literal

from distributed import get_worker

from rapidsmpf.config import Options
from rapidsmpf.integrations.core import (
    BCastJoinInfo,
    bcast_partition,
    bcast_shuffled_partition,
    get_allgather,
    get_new_shuffle_id,
    join_partition,
    stage_partitions,
)
from rapidsmpf.integrations.dask.core import (
    get_dask_client,
    get_dask_worker_rank,
    get_worker_context,
    global_rmpf_barrier,
)
from rapidsmpf.integrations.dask.shuffler import (
    _get_occupied_ids_dask,
    _shuffle_insertion_graph,
)

if TYPE_CHECKING:
    from distributed import Worker

    from rapidsmpf.integrations.core import JoinIntegration


def _stage_allgather(
    allgather_id: int,
    dask_worker: Worker | None = None,
) -> None:
    """
    Stage an allgather object without returning it.

    Parameters
    ----------
    allgather_id
        Unique ID for the allgather operation.
    dask_worker
        The current dask worker.

    Notes
    -----
    This function is expected to run on a Dask worker.
    """
    dask_worker = dask_worker or get_worker()
    get_allgather(
        get_worker_context(dask_worker),
        allgather_id,
        worker=dask_worker,
    )


def _worker_allgather_barrier(allgather_id: int, dependency: None) -> None:
    """
    Worker barrier for RapidsMPF allgather.

    Parameters
    ----------
    allgather_id
        Unique ID for the allgather operation.
    dependency
        Null argument used to enforce barrier dependencies.

    Notes
    -----
    A worker barrier task DOES need to be restricted
    to a specific Dask worker.
    """
    ctx = get_worker_context()
    with ctx.lock:
        get_allgather(ctx, allgather_id).insert_finished()


def rapidsmpf_join_graph(
    left_name: str,
    right_name: str,
    output_name: str,
    left_partition_count_in: int,
    right_partition_count_in: int,
    join_integration: JoinIntegration,
    left_options: Any,
    right_options: Any,
    join_options: Any,
    *,
    bcast_side: Literal["left", "right", None] = None,
    need_local_repartition: bool = False,
    left_pre_shuffled: bool = False,
    right_pre_shuffled: bool = False,
    config_options: Options = Options(),
) -> dict[Any, Any]:
    """
    Return the task graph for a RapidsMPF join.

    Parameters
    ----------
    left_name
        The name of the left table.
    right_name
        The name of the right table.
    output_name
        The name of the output table.
    left_partition_count_in
        The number of partitions in the left table.
    right_partition_count_in
        The number of partitions in the right table.
    join_integration
        The JoinIntegration protocol to use.
    left_options
        Additional options for extracting the left table.
    right_options
        Additional options for extracting the right table.
    join_options
        Additional options for the join.
    bcast_side
        The side of the join being broadcasted.
        Options are ``{'left', 'right', None}``.
        Note: Only ``None`` is supported for now.
    need_local_repartition
        Whether the join needs local repartitioning.
        The small-table will be shuffled (if it was not
        pre-shuffled) before being broadcasted, and each
        partition of the large table will be locally
        repartitioned before the join. This option
        should be set to ``False`` for inner joins.
    left_pre_shuffled
        Whether the left table is already shuffled.
    right_pre_shuffled
        Whether the right table is already shuffled.
    config_options
        RapidsMPF configuration options.

    Returns
    -------
    The task graph for the join operation.
    """
    # Get the Dask client and worker ranks
    client = get_dask_client(options=config_options)
    worker_ranks: dict[int, str] = {
        v: k for k, v in client.run(get_dask_worker_rank).items()
    }
    n_workers = len(worker_ranks)

    # Build the task-graph and restricted-key dicts incrementally
    restricted_keys: dict[Any, str] = {}
    graph: dict[Any, Any] = {}
    left_barrier_name: str | None = None
    right_barrier_name: str | None = None
    left_op_id: int | None = None
    right_op_id: int | None = None

    # Determine the number of partitions in the output table
    partition_count_out = max(left_partition_count_in, right_partition_count_in)

    if bcast_side is None:
        # Regular hash join

        # Shuffle left side (if necessary)
        if not left_pre_shuffled or left_partition_count_in != partition_count_out:
            (
                left_graph,
                left_barrier_name,
                left_restricted_keys,
                left_op_id,
            ) = _shuffle_insertion_graph(
                client,
                left_name,
                f"left-{output_name}",
                left_partition_count_in,
                partition_count_out,
                join_integration.get_shuffler_integration(),
                worker_ranks,
                left_options,
            )
            restricted_keys.update(left_restricted_keys)
            graph.update(left_graph)

        # Shuffle right side (if necessary)
        if not right_pre_shuffled or right_partition_count_in != partition_count_out:
            (
                right_graph,
                right_barrier_name,
                right_restricted_keys,
                right_op_id,
            ) = _shuffle_insertion_graph(
                client,
                right_name,
                f"right-{output_name}",
                right_partition_count_in,
                partition_count_out,
                join_integration.get_shuffler_integration(),
                worker_ranks,
                right_options,
            )
            restricted_keys.update(right_restricted_keys)
            graph.update(right_graph)

        # Add basic hash-join tasks
        for part_id in range(partition_count_out):
            rank = part_id % n_workers
            n_worker_tasks = partition_count_out // n_workers + int(
                rank < (partition_count_out % n_workers)
            )
            key = (output_name, part_id)
            graph[key] = (
                join_partition,
                get_worker_context,
                join_integration,
                None,  # Not a broadcast join
                left_op_id,
                right_op_id,
                left_barrier_name or (left_name, part_id),
                right_barrier_name or (right_name, part_id),
                part_id,
                n_worker_tasks,
                left_options,
                right_options,
                join_options,
            )
            # Assume round-robin partition assignment
            restricted_keys[key] = worker_ranks[rank]

    elif bcast_side in ["left", "right"]:
        # Get the operation id and stage the allgather operation
        allgather_id = get_new_shuffle_id(partial(_get_occupied_ids_dask, client))
        client.run(_stage_allgather, allgather_id)

        # Define task names for each phase of the broadcast join
        insert_name = f"bcast-insert-{output_name}"
        stage_name = f"bcast-stage-{output_name}"
        global_barrier_1_name = f"bcast-global-barrier-1-{output_name}"
        global_barrier_2_name = f"bcast-global-barrier-2-{output_name}"
        global_barrier_3_name = f"bcast-global-barrier-3-{output_name}"
        worker_barrier_name = f"bcast-worker-barrier-{output_name}"
        if bcast_side == "right":
            small_name = right_name
            small_count = right_partition_count_in
            bcast_options = right_options
        else:
            small_name = left_name
            small_count = left_partition_count_in
            bcast_options = left_options

        # Add tasks to broadcast each small-table partition
        insertion_keys: list[tuple[str, int]] = []
        if need_local_repartition and (
            (bcast_side == "right" and not right_pre_shuffled)
            or (bcast_side == "left" and not left_pre_shuffled)
        ):
            # Case #1: Pre-shuffle AND broadcast the small-table partitions
            (
                small_shuffle_graph,
                small_shuffle_barrier_name,
                small_shuffle_restricted_keys,
                small_shuffle_id,
            ) = _shuffle_insertion_graph(
                client,
                small_name,
                f"small-{output_name}",
                small_count,
                small_count,
                join_integration.get_shuffler_integration(),
                worker_ranks,
                bcast_options,
            )
            restricted_keys.update(small_shuffle_restricted_keys)
            graph.update(small_shuffle_graph)
            for pid in range(small_count):
                rank = pid % n_workers
                key = (insert_name, pid)
                graph[key] = (
                    bcast_shuffled_partition,
                    get_worker_context,
                    join_integration,
                    small_shuffle_id,
                    pid,
                    allgather_id,
                    bcast_options,
                    small_shuffle_barrier_name,
                )
                insertion_keys.append(key)
                # NOTE: We need to resttrict the keys to
                # specific workers to ensure we can extract
                # the correct partition for each index.
                restricted_keys[key] = worker_ranks[rank]
        else:
            # Case #2: Only broadcast the small-table partitions
            for pid in range(small_count):
                key = (insert_name, pid)
                graph[key] = (
                    bcast_partition,
                    get_worker_context,
                    join_integration,
                    (small_name, pid),
                    allgather_id,
                    bcast_options,
                )
                insertion_keys.append(key)

        # Add global barrier task
        graph[global_barrier_1_name] = (
            global_rmpf_barrier,
            *insertion_keys,
        )

        # Add worker barrier tasks
        worker_barriers: list[tuple[str, int]] = []
        for rank, addr in worker_ranks.items():
            key = (worker_barrier_name, rank)
            graph[key] = (
                _worker_allgather_barrier,
                allgather_id,
                global_barrier_1_name,
            )
            restricted_keys[key] = addr
            worker_barriers.append(key)

        # Add global barrier task
        graph[global_barrier_2_name] = (
            global_rmpf_barrier,
            *worker_barriers,
        )

        # Add staging tasks
        staging_tasks: list[tuple[str, int]] = []
        for rank, addr in worker_ranks.items():
            key = (stage_name, rank)
            graph[key] = (
                stage_partitions,
                get_worker_context,
                join_integration,
                allgather_id,
                bcast_options,
                global_barrier_2_name,
            )
            staging_tasks.append(key)
            restricted_keys[key] = addr

        # Add global barrier task
        graph[global_barrier_3_name] = (
            global_rmpf_barrier,
            *staging_tasks,
        )

        # Add join tasks
        bcast_info = BCastJoinInfo(
            bcast_side=bcast_side,
            bcast_count=small_count,
            need_local_repartition=need_local_repartition,
        )
        for part_id in range(partition_count_out):
            rank = part_id % n_workers
            n_worker_tasks = partition_count_out // n_workers + int(
                rank < (partition_count_out % n_workers)
            )
            key = (output_name, part_id)
            graph[key] = (
                join_partition,
                get_worker_context,
                join_integration,
                bcast_info,
                allgather_id if bcast_side == "left" else left_op_id,
                allgather_id if bcast_side == "right" else right_op_id,
                global_barrier_3_name if bcast_side == "left" else (left_name, part_id),
                global_barrier_3_name
                if bcast_side == "right"
                else (right_name, part_id),
                part_id,
                n_worker_tasks,
                left_options,
                right_options,
                join_options,
            )
            # Assume round-robin partition assignment
            restricted_keys[key] = worker_ranks[rank]

    else:  # pragma: no cover
        raise ValueError(f"Invalid bcast_side: {bcast_side}")

    # Tell the scheduler to restrict the worker-specific keys
    client._send_to_scheduler(
        {
            "op": "rmpf_add_restricted_tasks",
            "tasks": restricted_keys,
        }
    )

    # Return the full join task graph
    return graph
