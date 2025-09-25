# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Shuffler integration with external libraries."""

from __future__ import annotations

import threading
import weakref
from dataclasses import dataclass, field
from functools import cached_property, partial
from numbers import Number  # noqa: TC003
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Generic,
    Literal,
    Protocol,
    TypeVar,
    cast,
)

import rmm.mr
from rmm.pylibrmm.stream import DEFAULT_STREAM

from rapidsmpf.allgather import AllGather
from rapidsmpf.buffer.buffer import MemoryType
from rapidsmpf.buffer.resource import BufferResource, LimitAvailableMemory
from rapidsmpf.buffer.spill_collection import SpillCollection
from rapidsmpf.config import (
    Optional,
    OptionalBytes,
    Options,
)
from rapidsmpf.progress_thread import ProgressThread
from rapidsmpf.rmm_resource_adaptor import RmmResourceAdaptor
from rapidsmpf.shuffler import Shuffler
from rapidsmpf.statistics import Statistics

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from rapidsmpf.buffer.packed_data import PackedData
    from rapidsmpf.communicator.communicator import Communicator


DataFrameT = TypeVar("DataFrameT")


# Set of available shuffle IDs
_shuffle_id_vacancy: set[int] = set(range(Shuffler.max_concurrent_shuffles))
_shuffle_id_vacancy_lock: threading.Lock = threading.Lock()


def get_new_shuffle_id(get_occupied_ids: Callable[[], Sequence[set[int]]]) -> int:
    """
    Get a new available shuffle ID.

    Since RapidsMPF only supports a limited number of shuffler instances at
    any given time, this function maintains a shared pool of shuffle IDs.

    If no IDs are available locally, it calls get_occupied_ids to query all
    workers for IDs in use, updates the vacancy set accordingly, and retries.
    If all IDs are in use across all workers, an error is raised.

    Parameters
    ----------
    get_occupied_ids
        Callable function that returns the occupied shuffle IDs.

    Returns
    -------
    A unique shuffle ID not currently in use.

    Raises
    ------
    ValueError
        If all shuffle IDs are currently in use.
    """
    global _shuffle_id_vacancy  # noqa: PLW0603

    with _shuffle_id_vacancy_lock:
        if not _shuffle_id_vacancy:
            # We start with setting all IDs as vacant and then subtract all
            # IDs occupied on any one worker.
            _shuffle_id_vacancy = set(range(Shuffler.max_concurrent_shuffles))
            _shuffle_id_vacancy.difference_update(*get_occupied_ids())
            if not _shuffle_id_vacancy:
                raise ValueError(
                    f"Cannot shuffle more than {Shuffler.max_concurrent_shuffles} "
                    "times in a single query."
                )

        return _shuffle_id_vacancy.pop()


class StagedBcastData(Generic[DataFrameT]):
    """
    Staged broadcast data.

    Parameters
    ----------
    chunks
        The list of staged DataFrame chunks.
    """

    __slots__ = ("access_counts", "chunks")

    def __init__(self, chunks: list[DataFrameT]):
        self.chunks: list[DataFrameT] = chunks
        # We keep track of the number of times the chunks
        # have been accessed via the `get_chunk` method.
        # This is used to clean up the staged data after
        # a broadcast join is finished.
        self.access_counts: int = 0

    def get_chunk(self, chunk_id: int) -> DataFrameT:
        """
        Get a staged DataFrame chunk.

        Parameters
        ----------
        chunk_id
            The index of the chunk to return.

        Returns
        -------
        A DataFrame chunk.
        """
        self.access_counts += 1
        chunk: DataFrameT = self.chunks[chunk_id]
        return chunk


@dataclass
class WorkerContext:
    """
    RapidsMPF specific attributes for a worker.

    Attributes
    ----------
    lock
        The global worker lock. Must be acquired before accessing attributes
        that might be modified while the worker is running such as the shufflers.
    br
        The buffer resource used by the worker exclusively.
    progress_thread
        The progress thread used by the worker.
    comm
        The communicator connected to all other workers.
    statistics
        The statistics used by the worker. If None, statistics is disabled.
    spill_collection
        A collection of Python objects that can be spilled to free up device memory.
    shufflers
        A mapping from shuffler IDs to active shuffler or allgather instances.
    staged_bcast_data
        A mapping from allgather IDs to staged data.
        This attribute is used to stage data that has been broadcasted to the worker.
    options
        Configuration options.
    """

    lock: ClassVar[threading.RLock] = threading.RLock()
    br: BufferResource
    progress_thread: ProgressThread
    comm: Communicator
    statistics: Statistics
    spill_collection: SpillCollection = field(default_factory=SpillCollection)
    # TODO: Rename `shufflers`?
    shufflers: dict[int, Shuffler | AllGather] = field(default_factory=dict)
    staged_bcast_data: dict[int, StagedBcastData[Any]] = field(default_factory=dict)
    options: Options = field(default_factory=Options)

    def get_statistics(self) -> dict[str, dict[str, Number]]:
        """
        Get the statistics from the worker context.

        Returns
        -------
        statistics
            A dictionary of statistics. The keys are the names of the statistics.
            The values are dictionaries with two keys:

            - "count" is the number of times the statistic was recorded.
            - "value" is the value of the statistic.

        Notes
        -----
        Statistics are global across all shuffles. To measure statistics for any
        given shuffle, gather statistics before and after the shuffle and compute
        the difference.
        """
        return {
            stat: self.statistics.get_stat(stat)
            for stat in self.statistics.list_stat_names()
        }


class ShufflerIntegration(Protocol[DataFrameT]):
    """Shuffle-integration protocol."""

    @staticmethod
    def insert_partition(
        df: DataFrameT,
        partition_id: int,
        partition_count: int,
        shuffler: Shuffler,
        options: Any,
        *other: Any,
    ) -> None:
        """
        Add a partition to a RapidsMPF Shuffler.

        Parameters
        ----------
        df
            DataFrame partition to add to a RapidsMPF shuffler.
        partition_id
            The input partition id of ``df``.
        partition_count
            Number of output partitions for the current shuffle.
        shuffler
            The RapidsMPF Shuffler object to extract from.
        options
            Additional options.
        *other
            Other data needed for partitioning. For example,
            this may be boundary values needed for sorting.
        """
        ...

    @staticmethod
    def extract_partition(
        partition_id: int,
        shuffler: Shuffler,
        options: Any,
    ) -> DataFrameT:
        """
        Extract a DataFrame partition from a RapidsMPF Shuffler.

        Parameters
        ----------
        partition_id
            Partition id to extract.
        shuffler
            The RapidsMPF Shuffler object to extract from.
        options
            Additional options.

        Returns
        -------
        A shuffled DataFrame partition.
        """
        ...


def get_shuffler(
    ctx: WorkerContext,
    shuffle_id: int,
    *,
    partition_count: int | None = None,
    worker: Any = None,
) -> Shuffler:
    """
    Return the appropriate :class:`Shuffler` object.

    Parameters
    ----------
    ctx
        The worker context.
    shuffle_id
        Unique ID for the shuffle operation.
    partition_count
        Output partition count for the shuffle operation.
    worker
        The current worker.

    Returns
    -------
    The active RapidsMPF :class:`Shuffler` object associated with
    the specified ``shuffle_id``, ``partition_count`` and
    ``worker``.

    Notes
    -----
    Whenever a new :class:`Shuffler` object is created, it is
    saved as ``WorkerContext.shufflers[shuffle_id]``.
    """
    with ctx.lock:
        if shuffle_id not in ctx.shufflers:
            if partition_count is None:
                raise ValueError(
                    "Need partition_count to create new shuffler."
                    f" shuffle_id: {shuffle_id}\n"
                    f" Shufflers: {ctx.shufflers}"
                )
            assert ctx.br is not None
            assert ctx.comm is not None
            assert ctx.progress_thread is not None
            ctx.shufflers[shuffle_id] = Shuffler(
                ctx.comm,
                ctx.progress_thread,
                op_id=shuffle_id,
                total_num_partitions=partition_count,
                stream=DEFAULT_STREAM,
                br=ctx.br,
                statistics=ctx.statistics,
            )
        shuffler = ctx.shufflers[shuffle_id]
        assert isinstance(shuffler, Shuffler), (
            f"Found {type(shuffler)} at id={shuffle_id}. Expected Shuffler."
        )
        return shuffler


def _extract_shuffled_partition(
    get_worker_context: Callable[..., WorkerContext],
    shuffler_id: int,
    partition_id: int,
    shuffler_integration: ShufflerIntegration[DataFrameT],
    options: Any,
) -> DataFrameT:
    """
    Extract a shuffled partition from a Shuffler.

    Parameters
    ----------
    get_worker_context
        Callable function to fetch the worker context.
    shuffler_id
        The id of the Shuffler to extract from.
    partition_id
        The id of the partition to extract.
    shuffler_integration
        The ShufflerIntegration protocol to use.
    options
        Additional options.

    Returns
    -------
    A shuffled DataFrame partition.
    """
    ctx: WorkerContext = get_worker_context()
    shuffler = get_shuffler(ctx, shuffler_id)

    try:
        return shuffler_integration.extract_partition(partition_id, shuffler, options)
    finally:
        if shuffler.finished():
            with ctx.lock:
                if shuffler_id in ctx.shufflers:
                    del ctx.shufflers[shuffler_id]


def get_allgather(
    ctx: WorkerContext,
    allgather_id: int,
    *,
    worker: Any = None,
) -> AllGather:
    """
    Return the appropriate :class:`AllGather` object.

    Parameters
    ----------
    ctx
        The worker context.
    allgather_id
        Unique ID for the allgather operation.
    worker
        The current worker.

    Returns
    -------
    The active RapidsMPF :class:`AllGather` object associated with
    the specified ``allgather_id`` and ``worker``.

    Notes
    -----
    Whenever a new :class:`AllGather` object is created, it is
    saved as ``WorkerContext.allgathers[allgather_id]``.
    """
    with ctx.lock:
        if allgather_id not in ctx.shufflers:
            assert ctx.br is not None
            assert ctx.comm is not None
            assert ctx.progress_thread is not None
            ctx.shufflers[allgather_id] = AllGather(
                ctx.comm,
                ctx.progress_thread,
                op_id=allgather_id,
                stream=DEFAULT_STREAM,
                br=ctx.br,
                statistics=ctx.statistics,
            )
        allgather = ctx.shufflers[allgather_id]
        assert isinstance(allgather, AllGather), (
            f"Found {type(allgather)} at id={allgather_id}. Expected AllGather."
        )
        return allgather


def insert_partition(
    get_worker_context: Callable[..., WorkerContext],
    callback: Callable[
        [
            DataFrameT,
            int,
            int,
            Shuffler,
            Any,
            *tuple[str | tuple[str, int], ...],
        ],
        None,
    ],
    df: DataFrameT,
    partition_id: int,
    partition_count: int,
    shuffle_id: int,
    options: Any,
    *other_keys: str | tuple[str, int],
) -> None:
    """
    Add a partition to a RapidsMPF Shuffler.

    Parameters
    ----------
    get_worker_context
        Callable function to fetch the worker context.
    callback
        Insertion callback function. This function must be the
        `insert_partition` attribute of a `ShufflerIntegration`
        protocol.
    df
        DataFrame partition to add to a RapidsMPF shuffler.
    partition_id
        The input partition id of ``df``.
    partition_count
        Number of output partitions for the current shuffle.
    shuffle_id
        The RapidsMPF shuffle id.
    options
        Optional key-word arguments.
    *other_keys
        Other keys needed by ``callback``.
    """
    callback(
        df,
        partition_id,
        partition_count,
        get_shuffler(get_worker_context(), shuffle_id),
        options,
        *other_keys,
    )


def extract_partition(
    get_worker_context: Callable[..., WorkerContext],
    shuffler_integration: ShufflerIntegration[DataFrameT],
    shuffle_id: int,
    partition_id: int,
    worker_barrier: tuple[int, ...],
    options: Any,
) -> DataFrameT:
    """
    Extract a partition from a RapidsMPF Shuffler.

    Parameters
    ----------
    get_worker_context
        Callable function to fetch the worker context.
    shuffler_integration
        The ShufflerIntegration protocol to use.
    shuffle_id
        The RapidsMPF shuffle id.
    partition_id
        Partition id to extract.
    worker_barrier
        Worker-barrier task dependency. This value should
        not be used for compute logic.
    options
        Additional options.

    Returns
    -------
    Extracted DataFrame partition.
    """
    return _extract_shuffled_partition(
        get_worker_context,
        shuffle_id,
        partition_id,
        shuffler_integration,
        options,
    )


@dataclass
class BCastJoinInfo:
    """
    Broadcast join information.

    Parameters
    ----------
    bcast_side
        The side of the join being broadcasted.
    bcast_count
        The number of broadcasted partitions.
    need_local_repartition
        Whether to locally repartition on the broadcasted table.
        This is not necessary for inner joins.
    """

    bcast_side: Literal["left", "right"]
    bcast_count: int = 1
    need_local_repartition: bool = False


class JoinIntegration(Protocol[DataFrameT]):
    """Join-integration protocol."""

    @staticmethod
    def get_shuffler_integration() -> ShufflerIntegration[DataFrameT]:
        """Return the shuffler integration."""
        ...

    @staticmethod
    def pack_partition(
        ctx: WorkerContext, partition: DataFrameT, options: Any
    ) -> PackedData:
        """
        Pack a partition for broadcasting.

        Parameters
        ----------
        ctx
            The worker context.
        partition
            The partition to pack.
        options
            Additional options.

        Returns
        -------
        A PackedData object.
        """
        ...

    @staticmethod
    def unpack_partition(
        ctx: WorkerContext, data: PackedData, options: Any
    ) -> DataFrameT:
        """
        Unpack a broadcasted partition.

        Parameters
        ----------
        ctx
            The worker context.
        data
            The PackedData object to unpack.
        options
            Additional options.

        Returns
        -------
        A DataFrame partition.
        """
        ...

    @staticmethod
    def local_repartition(
        data: DataFrameT,
        partition_count: int,
        options: Any,
    ) -> dict[int, DataFrameT]:
        """
        Break a single DataFrame partition into multiple chunks.

        Parameters
        ----------
        data
            The local DataFrame partition.
        partition_count
            The number of local chunks to generate.
        options
            Additional options.

        Returns
        -------
        A dictionary of DataFrame chunks.
        The keys are the chunk ids.
        The values are the DataFrame chunks.
        """
        ...

    @staticmethod
    def join_partition(
        left_input: Callable[[int], DataFrameT],
        right_input: Callable[[int], DataFrameT],
        bcast_info: BCastJoinInfo | None,
        options: Any,
    ) -> DataFrameT:
        """
        Produce a joined table partition.

        Parameters
        ----------
        left_input
            A callable that produces chunks of the left partition.
            The ``bcast_info.bcast_count`` parameter corresponds
            to the number of chunks the callable can produce.
        right_input
            A callable that produces chunks of the right partition.
            The ``bcast_info.bcast_count`` parameter corresponds
            to the number of chunks the callable can produce.
        bcast_info
            The broadcast join information. This should be None
            for a regular hash join.
        options
            Additional join options.

        Returns
        -------
        A joined DataFrame partition.
        """
        ...


def bcast_partition(
    get_worker_context: Callable[..., WorkerContext],
    join_integration: JoinIntegration[DataFrameT],
    partition: DataFrameT,
    allgather_id: int,
    options: Any,
) -> None:
    """
    Broadcast a partition to all workers.

    Parameters
    ----------
    get_worker_context
        Callable function to fetch the worker context.
    join_integration
        The JoinIntegration protocol to use.
    partition
        The partition to broadcast.
    allgather_id
        The allgather id.
    options
        Additional options.
    """
    ctx = get_worker_context()
    with ctx.lock:
        allgather = get_allgather(ctx, allgather_id)
        allgather.insert(join_integration.pack_partition(ctx, partition, options))


def bcast_shuffled_partition(
    get_worker_context: Callable[..., WorkerContext],
    join_integration: JoinIntegration[DataFrameT],
    shuffler_id: int,
    partition_id: int,
    allgather_id: int,
    options: Any,
    barrier: Any,
) -> None:
    """
    Broadcast a shuffled partition to all workers.

    Parameters
    ----------
    get_worker_context
        Callable function to fetch the worker context.
    join_integration
        The JoinIntegration protocol to use.
    shuffler_id
        The id of the Shuffler providing the input partition.
    partition_id
        The input partition id.
    allgather_id
        The allgather id.
    options
        Additional options.
    barrier
        The global shuffle barrier.

    Notes
    -----
    This function differs from `bcast_partition`, because
    it depends on a Shuffler instance.
    """
    # Extract the input partition from the Shuffler
    partition = _extract_shuffled_partition(
        get_worker_context,
        shuffler_id,
        partition_id,
        join_integration.get_shuffler_integration(),
        options,
    )

    # Broadcast the partition to all workers
    bcast_partition(
        get_worker_context,
        join_integration,
        partition,
        allgather_id,
        options,
    )


def stage_partitions(
    get_worker_context: Callable[..., WorkerContext],
    integration: JoinIntegration[DataFrameT],
    allgather_id: int,
    options: Any,
    barrier: Any,
) -> None:
    """
    Stage broadcasted partitions for a broadcast join.

    Parameters
    ----------
    get_worker_context
        Callable function to fetch the worker context.
    integration
        The JoinIntegration protocol to use.
    allgather_id
        The allgather id.
    options
        Additional options.
    barrier
        The global broadcast join barrier.
    """
    ctx = get_worker_context()
    with ctx.lock:
        allgather = get_allgather(ctx, allgather_id)
        ordered = options.get("ordered", True)
        ctx.staged_bcast_data[allgather_id] = cast(
            "StagedBcastData[Any]",
            StagedBcastData(
                chunks=[
                    integration.unpack_partition(ctx, data, options)
                    for data in allgather.wait_and_extract(ordered=ordered)
                ]
            ),
        )
        for chunk in ctx.staged_bcast_data[allgather_id].chunks:
            # Make sure the staged data is spillable
            ctx.spill_collection.add_spillable(chunk)


class FetchJoinChunk(Generic[DataFrameT]):
    """
    Fetch the data for one side of a join operation.

    Parameters
    ----------
    side
        The side of the join being fetched.
    output_partition_id
        The output partition id for the join operation.
    get_worker_context
        Callable function to fetch the worker context.
    join_integration
        The JoinIntegration protocol to use.
    op_id
        The operation id.
    barrier
        The barrier to fetch the partition from.
    bcast_info
        The broadcast join information.
    n_worker_tasks
        The number of join_partition tasks to be called on this worker.
    options
        Additional options.

    Notes
    -----
    A ``FetchJoinChunk`` object only fetches data needed for a single
    output partition. For in-memory or shuffled data, there will only be
    one chunk to return. For broadcast joins, there may be multiple chunks.
    """

    def __init__(
        self,
        side: Literal["left", "right"],
        output_partition_id: int,
        get_worker_context: Callable[..., WorkerContext],
        join_integration: JoinIntegration[DataFrameT],
        op_id: int | None,
        barrier: DataFrameT | tuple[int, ...],
        bcast_info: BCastJoinInfo | None,
        n_worker_tasks: int,
        options: Any,
    ):
        self.side = side
        self.output_partition_id = output_partition_id
        self.get_worker_context = get_worker_context
        self.join_integration = join_integration
        self.op_id = op_id
        self.barrier = barrier
        self.bcast_info = bcast_info
        self.n_worker_tasks = n_worker_tasks
        self.options = options

    @cached_property
    def _data(self) -> dict[int, DataFrameT]:
        """Return a dictionary of DataFrame chunks."""
        data: DataFrameT
        if self.op_id is None:
            assert not isinstance(self.barrier, tuple), "Barrier must be a DataFrame."
            data = self.barrier
        else:
            data = _extract_shuffled_partition(
                self.get_worker_context,
                self.op_id,
                self.output_partition_id,
                self.join_integration.get_shuffler_integration(),
                self.options,
            )

        if (
            self.bcast_info is None
            or self.bcast_info.bcast_count == 1
            or not self.bcast_info.need_local_repartition
        ):
            return {0: data}
        else:
            return self.join_integration.local_repartition(
                data, self.bcast_info.bcast_count, self.options
            )

    def _get_bcasted_chunk(self, chunk_id: int) -> DataFrameT:
        """Return a broadcasted DataFrame chunk."""
        assert self.bcast_info is not None, "Broadcast join information is required."
        if chunk_id >= self.bcast_info.bcast_count:
            raise ValueError(
                f"Partition id {chunk_id} is out of bounds. Expected < {self.bcast_info.bcast_count}."
            )

        allgather_id = self.op_id
        access_count_limit = self.bcast_info.bcast_count * self.n_worker_tasks
        ctx = self.get_worker_context()
        with ctx.lock:
            if allgather_id not in ctx.staged_bcast_data:
                raise ValueError(
                    f"Allgather id {allgather_id} not found in staged data."
                )
            try:
                return cast(
                    "DataFrameT",
                    ctx.staged_bcast_data[allgather_id].get_chunk(chunk_id),
                )
            finally:
                # Cleanup
                if (
                    ctx.staged_bcast_data[allgather_id].access_counts
                    >= access_count_limit
                ):
                    assert ctx.shufflers[allgather_id].finished(), (
                        f"Allgather {allgather_id} is not finished."
                    )
                    del ctx.shufflers[allgather_id]
                    del ctx.staged_bcast_data[allgather_id]

    def __call__(self, chunk_id: int) -> Any:
        """
        Return the DataFrame associated with the given chunk id.

        Parameters
        ----------
        chunk_id
            The id of the local chunk to fetch for a join operation.
            There will only be one chunk to return for a hash join.
            There may be multiple chunks to return for a broadcast join.

        Returns
        -------
        A DataFrame chunk to be used in a join operation.
        """
        if self.bcast_info is None:
            # Fetch a chunk/partition for a regular hash join.
            # The chunk_id is ignored, because we only have a single chunk.
            return self._data[0]
        elif self.bcast_info.bcast_side == self.side:
            # Fetch a broadcasted chunk for a broadcast join.
            return self._get_bcasted_chunk(chunk_id)
        else:
            # Fetch a chunk of the un-broadcasted side of a broadcast join.
            return self._data[
                # Use the chunk_id if the data was locally repartitioned.
                chunk_id if self.bcast_info.need_local_repartition else 0
            ]


def join_partition(
    get_worker_context: Callable[..., WorkerContext],
    join_integration: JoinIntegration[DataFrameT],
    bcast_info: BCastJoinInfo | None,
    left_op_id: int | None,
    right_op_id: int | None,
    left_dependency: DataFrameT | tuple[int, ...],
    right_dependency: DataFrameT | tuple[int, ...],
    part_id: int,
    n_worker_tasks: int,
    left_options: Any,
    right_options: Any,
    join_options: Any,
) -> DataFrameT:
    """
    Produce a joined table partition.

    Parameters
    ----------
    get_worker_context
        Callable function to fetch the worker context.
    join_integration
        The JoinIntegration protocol to use.
    bcast_info
        The broadcast join information.
        This should be None for a regular hash join.
    left_op_id
        The left-table operation id. The operation may correspond
        to an allgather or a shuffle operation. If None, the
        left_dependency argument must be the left partition.
    right_op_id
        The right-table operation id. The operation may correspond
        to an allgather or a shuffle operation. If None, the
        right_dependency argument must be the right partition.
    left_dependency
        Task dependency for the left table. If left_op_id is None,
        this will correspond to the left partition. Otherwise, this
        argument is only used to enforce task ordering. The left_op_id
        argument should be used to fetch the real left partition.
    right_dependency
        Task dependency for the right table. If right_op_id is None,
        this will correspond to the right partition. Otherwise, this
        argument is only used to enforce task ordering. The right_op_id
        argument should be used to fetch the real right partition.
    part_id
        The output partition id.
        This information is needed to extract shuffled partitions.
    n_worker_tasks
        The number of join_partition tasks to be called on this worker.
        This information may be used for cleanup.
    left_options
        Additional options for extracting the left table.
    right_options
        Additional options for extracting the right table.
    join_options
        Additional options for the join.

    Returns
    -------
    A joined DataFrame partition.
    """

    def _get_input(side: Literal["left", "right"]) -> FetchJoinChunk:
        """Return the input for one side of the join."""
        if side == "left":
            op_id = left_op_id
            barrier = left_dependency
            options = left_options
        elif side == "right":
            op_id = right_op_id
            barrier = right_dependency
            options = right_options
        else:
            raise ValueError(f"Invalid side: {side}")

        return FetchJoinChunk(
            side,
            part_id,
            get_worker_context,
            join_integration,
            op_id,
            barrier,
            bcast_info,
            n_worker_tasks,
            options,
        )

    return join_integration.join_partition(
        _get_input("left"),
        _get_input("right"),
        bcast_info,
        join_options,
    )


# Create a spill function that spills the python objects in the spill-
# collection. This way, we have a central place (the worker) to track
# and trigger spilling of python objects.
def spill_func(
    amount: int,
    *,
    staging_buffer: rmm.DeviceBuffer | None,
    lock: threading.Lock,
    mr: rmm.mr.DeviceMemoryResource,
    ctx: WorkerContext,
) -> int:
    """
    Spill a specified amount of data from the Python object spill collection.

    This function attempts to use a preallocated staging device buffer to
    spill Python objects from the spill collection. If the staging buffer
    is currently in use, it will fall back to spilling without it.

    Parameters
    ----------
    amount
        The amount of data to spill, in bytes.
    staging_buffer
        Optional buffer to stage data through.
    lock
        Lock to protect access.
    mr
        Memory resource for device allocations.
    ctx
        The worker context to spill from.

    Returns
    -------
    The actual amount of data spilled, in bytes.
    """
    if staging_buffer is not None and lock.acquire(blocking=False):
        try:
            return ctx.spill_collection.spill(
                amount,
                stream=DEFAULT_STREAM,
                device_mr=mr,
                staging_device_buffer=staging_buffer,
            )
        finally:
            lock.release()
    return ctx.spill_collection.spill(amount, stream=DEFAULT_STREAM, device_mr=mr)


def rmpf_worker_setup(
    worker: Any,
    option_prefix: str,
    *,
    comm: Communicator,
    options: Options,
) -> WorkerContext:
    """
    Attach RapidsMPF shuffling attributes to a worker process.

    Parameters
    ----------
    worker
        The current worker process.
    option_prefix
        Prefix for config-option names.
    comm
        Communicator for shufflers.
    options
        Configuration options.

    Returns
    -------
    WorkerContext
        New worker context set up for shuffling.

    Warnings
    --------
    This function creates a new RMM memory pool, and
    sets it as the current device resource.
    """
    # Insert RMM resource adaptor on top of the current RMM resource stack.
    mr = RmmResourceAdaptor(
        upstream_mr=rmm.mr.get_current_device_resource(),
        fallback_mr=(
            # Use a managed memory resource if OOM protection is enabled.
            rmm.mr.ManagedMemoryResource()
            if options.get_or_default(
                f"{option_prefix}oom_protection", default_value=False
            )
            else None
        ),
    )
    rmm.mr.set_current_device_resource(mr)

    # Print statistics at worker shutdown.
    if options.get_or_default(f"{option_prefix}statistics", default_value=False):
        statistics = Statistics(enable=True, mr=mr)
    else:
        statistics = Statistics(enable=False)

    if (
        options.get_or_default(f"{option_prefix}print_statistics", default_value=True)
        and statistics.enabled
    ):
        weakref.finalize(
            worker,
            lambda name, stats: print(name, stats.report()),
            name=str(worker),
            stats=statistics,
        )

    # Create a buffer resource with a limiting availability function.
    total_memory = rmm.mr.available_device_memory()[1]
    spill_device = options.get_or_default(
        f"{option_prefix}spill_device", default_value=0.50
    )
    memory_available = {
        MemoryType.DEVICE: LimitAvailableMemory(
            mr, limit=int(total_memory * spill_device)
        )
    }
    br = BufferResource(
        mr,
        memory_available=memory_available,
        periodic_spill_check=options.get_or_default(
            f"{option_prefix}periodic_spill_check", default_value=Optional(1e-3)
        ).value,
    )

    # If enabled, create a staging device buffer for the spilling to reduce
    # device memory pressure.
    # TODO: maybe have a pool of staging buffers?
    spill_staging_buffer_size = options.get_or_default(
        f"{option_prefix}staging_spill_buffer",
        default_value=OptionalBytes("128 MiB"),
    ).value
    spill_staging_buffer = (
        None
        if spill_staging_buffer_size is None
        else rmm.DeviceBuffer(
            size=spill_staging_buffer_size, stream=DEFAULT_STREAM, mr=mr
        )
    )
    ctx = WorkerContext(
        br=br,
        progress_thread=ProgressThread(comm, statistics),
        comm=comm,
        statistics=statistics,
        options=options,
    )

    # Add the spill function using a negative priority (-10) such that spilling
    # of internal shuffle buffers (non-python objects) have higher priority than
    # spilling of the Python objects in the collection.
    br.spill_manager.add_spill_function(
        func=partial(
            spill_func,
            staging_buffer=spill_staging_buffer,
            lock=threading.Lock(),
            mr=mr,
            ctx=ctx,
        ),
        priority=-10,
    )
    return ctx
