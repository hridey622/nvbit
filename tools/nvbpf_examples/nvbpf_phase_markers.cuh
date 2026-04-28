#pragma once

/*
 * Source-level semantic phase markers for megakernel debugging.
 *
 * Include this header in CUDA/CuTe/CUTLASS kernels and place BEGIN/END/MARK
 * macros around phases whose ordering you want NVBPF to prove. The marker is a
 * volatile inline-PTX MOV with a magic immediate. It has no memory side effects;
 * megakernel_phase_comm_proof.so recognizes the immediate in SASS and records a
 * timestamped phase event.
 *
 * The phase argument must be a compile-time constant from the list below.
 */

#define NVBPF_PHASE_MARKER_MAGIC 0x4e564200u

#define NVBPF_PHASE_ENTRY 0u
#define NVBPF_PHASE_TMA 1u
#define NVBPF_PHASE_TMEM 2u
#define NVBPF_PHASE_TENSOR 3u
#define NVBPF_PHASE_SCALE 4u
#define NVBPF_PHASE_MULTICAST 5u
#define NVBPF_PHASE_SYNC 6u
#define NVBPF_PHASE_PRODUCER 7u
#define NVBPF_PHASE_HANDOFF 8u
#define NVBPF_PHASE_CONSUMER 9u
#define NVBPF_PHASE_EPILOGUE 10u
#define NVBPF_PHASE_BRANCH 11u

#if defined(__CUDA_ARCH__)
#define NVBPF_PHASE_MARK(phase_id)                                      \
    do {                                                                \
        unsigned int _nvbpf_phase_marker_sink;                          \
        asm volatile("mov.u32 %0, %1;"                                  \
                     : "=r"(_nvbpf_phase_marker_sink)                   \
                     : "n"(NVBPF_PHASE_MARKER_MAGIC | ((phase_id)&0xffu)) \
                     : "memory");                                       \
        asm volatile("" : : "r"(_nvbpf_phase_marker_sink) : "memory");  \
    } while (0)
#else
#define NVBPF_PHASE_MARK(phase_id) do { (void)(phase_id); } while (0)
#endif

#define NVBPF_PHASE_BEGIN(phase_id) NVBPF_PHASE_MARK(phase_id)
#define NVBPF_PHASE_END(phase_id) NVBPF_PHASE_MARK(phase_id)

#define NVBPF_MARK_TMA() NVBPF_PHASE_MARK(NVBPF_PHASE_TMA)
#define NVBPF_MARK_TMEM() NVBPF_PHASE_MARK(NVBPF_PHASE_TMEM)
#define NVBPF_MARK_TENSOR() NVBPF_PHASE_MARK(NVBPF_PHASE_TENSOR)
#define NVBPF_MARK_SCALE() NVBPF_PHASE_MARK(NVBPF_PHASE_SCALE)
#define NVBPF_MARK_MULTICAST() NVBPF_PHASE_MARK(NVBPF_PHASE_MULTICAST)
#define NVBPF_MARK_SYNC() NVBPF_PHASE_MARK(NVBPF_PHASE_SYNC)
#define NVBPF_MARK_PRODUCER() NVBPF_PHASE_MARK(NVBPF_PHASE_PRODUCER)
#define NVBPF_MARK_HANDOFF() NVBPF_PHASE_MARK(NVBPF_PHASE_HANDOFF)
#define NVBPF_MARK_CONSUMER() NVBPF_PHASE_MARK(NVBPF_PHASE_CONSUMER)
#define NVBPF_MARK_EPILOGUE() NVBPF_PHASE_MARK(NVBPF_PHASE_EPILOGUE)
