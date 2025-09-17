# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Join integration for Dask Distributed clusters."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Any, Literal, Protocol, TypeVar

from rapidsmpf.config import Options
from rapidsmpf.integrations.core import (
    get_new_operation_id,
    insert_partition,
    join_chunk,
)
from rapidsmpf.integrations.dask.core import (
    get_dask_client,
    get_dask_worker_rank,
    get_worker_context,
    global_rmpf_barrier,
)
from rapidsmpf.integrations.dask.shuffler import (
    _get_occupied_ids_dask,
    _stage_shuffler,
    _worker_rmpf_barrier,
)

if TYPE_CHECKING:
    from distributed import Client

    from rapidsmpf.integrations.core import ShufflerIntegration


DataFrameT = TypeVar("DataFrameT")


def join_chunk(
    get_context: Callable[..., WorkerContext],
    callback: Callable[
        [WorkerContext, Literal["left", "right", "none"], int, int, Any],
        DataFrameT,
    ],
    bcast_side: Literal["left", "right", "none"],
    left_op_id: int,
    right_op_id: int,
    left_barrier: tuple[int, ...],
    right_barrier: tuple[int, ...],
    part_id: int,
    options: Any,
) -> DataFrameT:
    ctx = get_context()

    return callback(
        ctx,
        bcast_side,
        left_op_id,
        right_op_id,
        part_id,
        options,
    )


class JoinIntegration(Protocol[DataFrameT]):
    """Join-integration protocol."""

    shuffle_integration: ShufflerIntegration[DataFrameT]

    @staticmethod
    def bcast_chunk(
        df: DataFrameT,
        allgather: AllGather,
        options: Any,
    ) -> None:
        """
        Broadcast a DataFrame chunk to all workers.

        Parameters
        ----------
        df
            DataFrame partition to add to a RapidsMPF shuffler.
        allgather
            The RapidsMPF AllGather object to use.
        options
            Additional options.

        Notes
        -----
        This method is used to broadcast a small-table DataFrame
        chunk to all workers during a broadcast join.
        """
        ...

    @staticmethod
    def stage_bcast_results(
        ctx: WorkerContext,
        allgather_id: int,
        options: Any,
    ) -> int:
        """
        Stage the results of a RapidsMPF AllGather broadcast.

        Parameters
        ----------
        ctx
            The worker context.
        allgather_id
            The AllGather operation id.
        options
            Additional options.

        Returns
        -------
        The number of staged chunks associated with allgather_id.

        Notes
        -----
        This method is used to stage the broadcasted small-table
        chunks on each worker during a broadcast join. This data
        should be added to the workers SpillCollection and stored
        in ctx.staged_results[allgather_id].
        """
        ...

    @classmethod
    def join_chunk(
        cls,
        ctx: WorkerContext,
        bcast_side: Literal["left", "right", "none"],
        left_op_id: int,
        right_op_id: int,
        part_id: int,
        options: Any,
    ) -> DataFrameT:
        """
        Perform a join operation on left and right table chunks.

        Parameters
        ----------
        ctx
            The worker context.
        bcast_side
            The side of the join being broadcasted. If "none", this is
            a regular hash join.
        left_op_id
            The left-table operation id. The operation may correspond
            to an allgather or shuffle operation.
        right_op_id
            The right-table operation id. The operation may correspond
            to an allgather or shuffle operation.
        part_id
            The output partition id.
        options
            Additional options.

        Returns
        -------
        A joined DataFrame chunk.

        Notes
        -----
        This method is used to produce a single joined table chunk.
        """
        ...


def _shuffle_graph(
    client: Client,
    input_name: str,
    output_name: str,
    partition_count_in: int,
    partition_count_out: int,
    integration: ShufflerIntegration,
    worker_ranks: dict[int, str],
    options: Any,
) -> tuple[int, dict[Any, str], dict[Any, Any]]:
    """Shuffle one table involved for a join."""
    # Get the operation id
    shuffle_id = get_new_operation_id(partial(_get_occupied_ids_dask, client))
    client.run(_stage_shuffler, shuffle_id, partition_count_out)
    n_workers = len(worker_ranks)
    restricted_keys: dict[Any, str] = {}

    # Define task names for each phase of the shuffle
    insert_name = f"rmpf-insert-{output_name}"
    global_barrier_name = f"rmpf-global-shuffle-barrier-{output_name}"
    worker_barrier_name = f"rmpf-worker-barrier-{output_name}"

    # Add tasks to insert each partition into the shuffler
    graph: dict[Any, Any] = {
        (insert_name, pid): (
            insert_partition,
            get_worker_context,
            integration.insert_partition,
            (input_name, pid),
            pid,
            partition_count_out,
            shuffle_id,
            options,
        )
        for pid in range(partition_count_in)
    }

    # Add global barrier task
    graph[global_barrier_name] = (
        global_rmpf_barrier,
        *graph.keys(),
    )

    # Add worker barrier tasks
    worker_barriers: dict[Any, Any] = {}
    for rank, addr in worker_ranks.items():
        key = (worker_barrier_name, rank)
        worker_barriers[rank] = key
        graph[key] = (
            _worker_rmpf_barrier,
            (shuffle_id,),
            partition_count_out,
            global_barrier_name,
        )
        restricted_keys[key] = addr

    # Add global barrier task
    graph[output_name] = (
        global_rmpf_barrier,
        *worker_barriers.values(),
    )

    return shuffle_id, restricted_keys, graph


def rapidsmpf_join_graph(
    left_name: str,
    right_name: str,
    output_name: str,
    bcast_side: Literal["left", "right", "none"],
    left_partition_count_in: int,
    right_partition_count_in: int,
    integration: JoinIntegration,
    options: Any,
    *,
    shuffle_small: bool = False,
    config_options: Options = Options(),
) -> dict[Any, Any]:
    """Return the task graph for a RapidsMPF broadcast join."""
    client = get_dask_client(options=config_options)
    worker_ranks: dict[int, str] = {
        v: k for k, v in client.run(get_dask_worker_rank).items()
    }
    n_workers = len(worker_ranks)
    restricted_keys: dict[Any, str] = {}
    graph: dict[Any, Any] = {}

    if bcast_side == "none":
        # Regular hash join
        partition_count_out = max(left_partition_count_in, right_partition_count_in)

        # TODO: What if one or both sides is already shuffled?
        # Perhaps the user shouldn't use this function in that
        # case, but we may be able to handle it here.

        # Shuffle left side
        left_barrier_name = f"rmpf-shuffle-left-{output_name}"
        left_op_id, left_restricted_keys, left_graph = _shuffle_graph(
            client,
            left_name,
            left_barrier_name,
            left_partition_count_in,
            partition_count_out,
            integration.shuffle_integration,
            worker_ranks,
            {"on": options["left_on"]},
        )
        restricted_keys.update(left_restricted_keys)
        graph.update(left_graph)

        # Shuffle right side
        right_barrier_name = f"rmpf-shuffle-right-{output_name}"
        right_op_id, right_restricted_keys, right_graph = _shuffle_graph(
            client,
            right_name,
            right_barrier_name,
            right_partition_count_in,
            partition_count_out,
            integration.shuffle_integration,
            worker_ranks,
            {"on": options["right_on"]},
        )
        restricted_keys.update(right_restricted_keys)
        graph.update(right_graph)

    elif bcast_side in ["left", "right"]:
        # Broadcast join

        raise NotImplementedError("Broadcast join not implemented.")

        if bcast_side == "right":
            small_name = right_name
            large_name = left_name
            small_count = right_partition_count_in
            large_count = left_partition_count_in
        else:
            small_name = left_name
            large_name = right_name
            small_count = left_partition_count_in
            large_count = right_partition_count_in

        if shuffle_small:
            small_shuffle_name = f"rmpf-shuffle-small-{output_name}"
            small_shuffle_id, small_restricted_keys, small_graph = _shuffle_graph(
                client,
                small_name,
                small_shuffle_name,
                small_count,
                large_count,
                integration.shuffle_integration,
                options,
            )
            restricted_keys.update(small_restricted_keys)

    else:  # pragma: no cover
        raise ValueError(f"Invalid broadcast side: {bcast_side}")

    # Add basic hash-join tasks
    for part_id in range(partition_count_out):
        rank = part_id % n_workers
        key = (output_name, part_id)
        graph[key] = (
            join_chunk,
            get_worker_context,
            integration.join_chunk,
            bcast_side,
            left_op_id,
            right_op_id,
            left_barrier_name,
            right_barrier_name,
            part_id,
            options,
        )
        # Assume round-robin partition assignment
        restricted_keys[key] = worker_ranks[rank]

    # Tell the scheduler to restrict the worker-specific keys
    client._send_to_scheduler(
        {
            "op": "rmpf_add_restricted_tasks",
            "tasks": restricted_keys,
        }
    )

    return graph
