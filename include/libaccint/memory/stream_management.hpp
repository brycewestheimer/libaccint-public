// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

#pragma once

/// @file stream_management.hpp
/// @brief CUDA stream management utilities for GPU backends
///
/// Provides RAII wrappers for CUDA streams and events, plus a stream pool
/// for concurrent kernel execution.

#include <libaccint/config.hpp>
#include <libaccint/memory/device_memory.hpp>

#include <condition_variable>
#include <cstddef>
#include <memory>
#include <mutex>
#include <queue>
#include <vector>

#if LIBACCINT_USE_CUDA
#include <cuda_runtime.h>
#endif

namespace libaccint::memory {

#if LIBACCINT_USE_CUDA

using gpu_stream_t = cudaStream_t;
using gpu_event_t = cudaEvent_t;

// ============================================================================
// StreamHandle - RAII wrapper for cudaStream_t
// ============================================================================

/**
 * @brief RAII wrapper for a CUDA stream
 *
 * Automatically creates a stream on construction and destroys it on
 * destruction. Move-only to prevent double-free.
 *
 * Example usage:
 * @code
 *     StreamHandle stream;
 *     my_kernel<<<blocks, threads, 0, stream.get()>>>(args...);
 *     stream.synchronize();
 * @endcode
 */
// Default flag constants for stream/event creation
inline constexpr unsigned int kGpuStreamNonBlocking = cudaStreamNonBlocking;
inline constexpr unsigned int kGpuEventDefault = cudaEventDefault;

class StreamHandle {
public:
    /**
     * @brief Construct a new stream
     * @param flags Stream creation flags (default: non-blocking)
     * @throws CudaError on stream creation failure
     */
    explicit StreamHandle(unsigned int flags = kGpuStreamNonBlocking);

    /// Destructor - destroys the stream
    ~StreamHandle();

    // Move constructor
    StreamHandle(StreamHandle&& other) noexcept;

    // Move assignment
    StreamHandle& operator=(StreamHandle&& other) noexcept;

    // Delete copy operations
    StreamHandle(const StreamHandle&) = delete;
    StreamHandle& operator=(const StreamHandle&) = delete;

    // ---- Accessors ----

    /// Get the underlying GPU stream
    [[nodiscard]] gpu_stream_t get() const noexcept { return stream_; }

    /// Implicit conversion to gpu_stream_t for convenience
    operator gpu_stream_t() const noexcept { return stream_; }

    /// Check if this handle owns a valid stream
    [[nodiscard]] bool valid() const noexcept { return stream_ != nullptr; }

    // ---- Operations ----

    /**
     * @brief Synchronize this stream (wait for all operations to complete)
     * @throws CudaError on synchronization failure
     */
    void synchronize();

    /**
     * @brief Query if all operations in this stream are complete
     * @return true if stream is idle, false if operations are pending
     */
    [[nodiscard]] bool query() const;

    /**
     * @brief Make this stream wait for an event
     * @param event The event to wait for
     * @throws CudaError on failure
     */
    void wait_event(gpu_event_t event);

private:
    gpu_stream_t stream_{nullptr};
};

// ============================================================================
// EventHandle - RAII wrapper for cudaEvent_t
// ============================================================================

/**
 * @brief RAII wrapper for a CUDA event
 *
 * Automatically creates an event on construction and destroys it on
 * destruction. Move-only to prevent double-free.
 */
class EventHandle {
public:
    /**
     * @brief Construct a new event
     * @param flags CUDA event creation flags (default: cudaEventDefault)
     * @throws CudaError on event creation failure
     */
    explicit EventHandle(unsigned int flags = kGpuEventDefault);

    /// Destructor - destroys the event
    ~EventHandle();

    // Move constructor
    EventHandle(EventHandle&& other) noexcept;

    // Move assignment
    EventHandle& operator=(EventHandle&& other) noexcept;

    // Delete copy operations
    EventHandle(const EventHandle&) = delete;
    EventHandle& operator=(const EventHandle&) = delete;

    // ---- Accessors ----

    /// Get the underlying GPU event
    [[nodiscard]] gpu_event_t get() const noexcept { return event_; }

    /// Implicit conversion to gpu_event_t for convenience
    operator gpu_event_t() const noexcept { return event_; }

    /// Check if this handle owns a valid event
    [[nodiscard]] bool valid() const noexcept { return event_ != nullptr; }

    // ---- Operations ----

    /**
     * @brief Record this event in a stream
     * @param stream The stream to record in (nullptr = default stream)
     * @throws CudaError on failure
     */
    void record(gpu_stream_t stream = nullptr);

    /**
     * @brief Synchronize on this event (wait for it to complete)
     * @throws CudaError on failure
     */
    void synchronize();

    /**
     * @brief Query if this event has completed
     * @return true if event has completed, false if still pending
     */
    [[nodiscard]] bool query() const;

private:
    gpu_event_t event_{nullptr};
};

// ============================================================================
// EventTimer - Timing utilities with CUDA events
// ============================================================================

/**
 * @brief GPU timing utility using CUDA events
 *
 * Provides accurate timing of GPU operations by recording events before
 * and after the operations of interest.
 *
 * Example usage:
 * @code
 *     EventTimer timer;
 *     timer.start(stream);
 *     // ... kernel launches ...
 *     timer.stop(stream);
 *     cudaStreamSynchronize(stream);
 *     float ms = timer.elapsed_ms();
 * @endcode
 */
class EventTimer {
public:
    /**
     * @brief Construct a new timer with default timing events
     * @throws CudaError on event creation failure
     */
    EventTimer();

    /// Destructor
    ~EventTimer() = default;

    // Move operations
    EventTimer(EventTimer&&) = default;
    EventTimer& operator=(EventTimer&&) = default;

    // Delete copy operations
    EventTimer(const EventTimer&) = delete;
    EventTimer& operator=(const EventTimer&) = delete;

    /**
     * @brief Record the start time in a stream
     * @param stream The stream to record in (nullptr = default stream)
     * @throws CudaError on failure
     */
    void start(gpu_stream_t stream = nullptr);

    /**
     * @brief Record the stop time in a stream
     * @param stream The stream to record in (nullptr = default stream)
     * @throws CudaError on failure
     */
    void stop(gpu_stream_t stream = nullptr);

    /**
     * @brief Get elapsed time between start and stop events
     * @return Elapsed time in milliseconds
     * @throws CudaError if events haven't completed or weren't recorded
     * @note Both start() and stop() must have been called, and the stream
     *       must be synchronized before calling this.
     */
    [[nodiscard]] float elapsed_ms() const;

private:
    EventHandle start_event_;
    EventHandle stop_event_;
    bool started_{false};
    bool stopped_{false};
};

// ============================================================================
// StreamPool - Pool of CUDA streams for concurrent execution
// ============================================================================

/**
 * @brief Pool of CUDA streams for concurrent kernel execution
 *
 * Maintains a pool of streams that can be acquired and released for
 * concurrent kernel launches. Thread-safe for multi-threaded host code.
 *
 * Example usage:
 * @code
 *     StreamPool pool(4);  // 4 streams
 *
 *     // Acquire streams for concurrent work
 *     auto& stream1 = pool.acquire();
 *     auto& stream2 = pool.acquire();
 *
 *     kernel1<<<blocks, threads, 0, stream1.get()>>>(args1...);
 *     kernel2<<<blocks, threads, 0, stream2.get()>>>(args2...);
 *
 *     pool.release(stream1);
 *     pool.release(stream2);
 *
 *     pool.synchronize_all();
 * @endcode
 */
class StreamPool {
public:
    /**
     * @brief Construct a stream pool with the specified number of streams
     * @param n_streams Number of streams in the pool (default: 4)
     * @param flags CUDA stream creation flags
     * @throws CudaError if stream creation fails
     */
    explicit StreamPool(size_t n_streams = 4,
                        unsigned int flags = kGpuStreamNonBlocking);

    /// Destructor - destroys all streams in the pool
    ~StreamPool() = default;

    // Non-copyable, non-movable (manages shared pool state)
    StreamPool(const StreamPool&) = delete;
    StreamPool& operator=(const StreamPool&) = delete;
    StreamPool(StreamPool&&) = delete;
    StreamPool& operator=(StreamPool&&) = delete;

    // ---- Pool Operations ----

    /**
     * @brief Acquire a stream from the pool
     *
     * If no streams are available, blocks until one is released.
     * The returned stream reference is valid until release() is called.
     *
     * @return Reference to an available stream
     * @note Thread-safe
     */
    [[nodiscard]] StreamHandle& acquire();

    /**
     * @brief Release a stream back to the pool
     * @param stream The stream to release (must have been acquired from this pool)
     * @note Thread-safe
     */
    void release(StreamHandle& stream);

    /**
     * @brief Synchronize all streams in the pool
     *
     * Waits for all operations in all streams to complete.
     * @throws CudaError on synchronization failure
     */
    void synchronize_all();

    /**
     * @brief Get the total number of streams in the pool
     * @return Number of streams
     */
    [[nodiscard]] size_t size() const noexcept { return streams_.size(); }

    /**
     * @brief Get the number of currently available streams
     * @return Number of streams not currently acquired
     * @note Thread-safe
     */
    [[nodiscard]] size_t available() const;

private:
    std::vector<std::unique_ptr<StreamHandle>> streams_;
    std::queue<StreamHandle*> available_;
    mutable std::mutex mutex_;
    std::condition_variable cv_;
};

// ============================================================================
// ScopedStream - RAII helper for automatic stream release
// ============================================================================

/**
 * @brief RAII helper that automatically releases a stream back to a pool
 *
 * Example usage:
 * @code
 *     StreamPool pool;
 *     {
 *         ScopedStream scoped(pool);
 *         kernel<<<blocks, threads, 0, scoped.get()>>>(args...);
 *     }  // Stream automatically released here
 * @endcode
 */
class ScopedStream {
public:
    /**
     * @brief Acquire a stream from a pool
     * @param pool The pool to acquire from
     */
    explicit ScopedStream(StreamPool& pool);

    /// Destructor - releases the stream back to the pool
    ~ScopedStream();

    // Non-copyable, non-movable
    ScopedStream(const ScopedStream&) = delete;
    ScopedStream& operator=(const ScopedStream&) = delete;
    ScopedStream(ScopedStream&&) = delete;
    ScopedStream& operator=(ScopedStream&&) = delete;

    /// Get the underlying stream
    [[nodiscard]] gpu_stream_t get() const noexcept { return stream_->get(); }

    /// Get the stream handle
    [[nodiscard]] StreamHandle& handle() noexcept { return *stream_; }

    /// Implicit conversion to gpu_stream_t
    operator gpu_stream_t() const noexcept { return stream_->get(); }

private:
    StreamPool& pool_;
    StreamHandle* stream_;
};

#endif  // LIBACCINT_USE_CUDA

}  // namespace libaccint::memory
