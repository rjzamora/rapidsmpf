# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from libc.stdint cimport uint8_t, uint32_t
from libcpp cimport bool
from libcpp.memory cimport shared_ptr, unique_ptr
from libcpp.string cimport string
from libcpp.unordered_map cimport unordered_map
from libcpp.vector cimport vector
from rmm.librmm.cuda_stream_view cimport cuda_stream_view
from rmm.pylibrmm.stream cimport Stream

from rapidsmpf.buffer.packed_data cimport cpp_PackedData
from rapidsmpf.buffer.resource cimport BufferResource, cpp_BufferResource
from rapidsmpf.communicator.communicator cimport Communicator, cpp_Communicator
from rapidsmpf.progress_thread cimport cpp_ProgressThread
from rapidsmpf.statistics cimport cpp_Statistics


cdef extern from "<rapidsmpf/allgather/allgather.hpp>" nogil:
    cdef cppclass cpp_AllGather "rapidsmpf::allgather::AllGather":
        cpp_AllGather(
            shared_ptr[cpp_Communicator] comm,
            shared_ptr[cpp_ProgressThread] comm,
            uint8_t op_id,
            cuda_stream_view stream,
            cpp_BufferResource *br,
            shared_ptr[cpp_Statistics] statistics,
        ) except +
        void insert(unordered_map[uint32_t, cpp_PackedData] chunks) except +
        void insert_finished(vector[uint32_t] pids) except +
        # TODO: Add methods to extract the results

cdef class AllGather:
    cdef unique_ptr[cpp_Shuffler] _handle
    cdef Communicator _comm
    cdef Stream _stream
    cdef BufferResource _br
