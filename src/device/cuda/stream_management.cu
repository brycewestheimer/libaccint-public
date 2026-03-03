// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file stream_management.cu
/// @brief CUDA stream management implementation

#include <libaccint/memory/stream_management.hpp>

#if LIBACCINT_USE_CUDA

namespace libaccint::memory {

// ============================================================================
// StreamHandle Implementation
// ============================================================================

StreamHandle::StreamHandle(unsigned int flags) : stream_(nullptr) {
    LIBACCINT_CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, flags));
}

StreamHandle::~StreamHandle() {
    if (stream_ != nullptr) {
        // Ignore errors during destruction to avoid exceptions in destructors
        cudaStreamDestroy(stream_);
    }
}

StreamHandle::StreamHandle(StreamHandle&& other) noexcept
    : stream_(other.stream_) {
    other.stream_ = nullptr;
}

StreamHandle& StreamHandle::operator=(StreamHandle&& other) noexcept {
    if (this != &other) {
        if (stream_ != nullptr) {
            cudaStreamDestroy(stream_);
        }
        stream_ = other.stream_;
        other.stream_ = nullptr;
    }
    return *this;
}

void StreamHandle::synchronize() {
    LIBACCINT_CUDA_CHECK(cudaStreamSynchronize(stream_));
}

bool StreamHandle::query() const {
    cudaError_t status = cudaStreamQuery(stream_);
    if (status == cudaSuccess) {
        return true;
    } else if (status == cudaErrorNotReady) {
        return false;
    }
    // Any other error is a real error - throw with proper error message
    throw CudaError(cudaGetErrorString(status), __FILE__, __LINE__);
}

void StreamHandle::wait_event(cudaEvent_t event) {
    LIBACCINT_CUDA_CHECK(cudaStreamWaitEvent(stream_, event, 0));
}

// ============================================================================
// EventHandle Implementation
// ============================================================================

EventHandle::EventHandle(unsigned int flags) {
    LIBACCINT_CUDA_CHECK(cudaEventCreateWithFlags(&event_, flags));
}

EventHandle::~EventHandle() {
    if (event_ != nullptr) {
        cudaEventDestroy(event_);
    }
}

EventHandle::EventHandle(EventHandle&& other) noexcept
    : event_(other.event_) {
    other.event_ = nullptr;
}

EventHandle& EventHandle::operator=(EventHandle&& other) noexcept {
    if (this != &other) {
        if (event_ != nullptr) {
            cudaEventDestroy(event_);
        }
        event_ = other.event_;
        other.event_ = nullptr;
    }
    return *this;
}

void EventHandle::record(cudaStream_t stream) {
    LIBACCINT_CUDA_CHECK(cudaEventRecord(event_, stream));
}

void EventHandle::synchronize() {
    LIBACCINT_CUDA_CHECK(cudaEventSynchronize(event_));
}

bool EventHandle::query() const {
    cudaError_t status = cudaEventQuery(event_);
    if (status == cudaSuccess) {
        return true;
    } else if (status == cudaErrorNotReady) {
        return false;
    }
    // Any other error is a real error - throw with proper error message
    throw CudaError(cudaGetErrorString(status), __FILE__, __LINE__);
}

// ============================================================================
// EventTimer Implementation
// ============================================================================

EventTimer::EventTimer()
    : start_event_(cudaEventDefault),
      stop_event_(cudaEventDefault) {
}

void EventTimer::start(cudaStream_t stream) {
    start_event_.record(stream);
    started_ = true;
    stopped_ = false;
}

void EventTimer::stop(cudaStream_t stream) {
    stop_event_.record(stream);
    stopped_ = true;
}

float EventTimer::elapsed_ms() const {
    if (!started_ || !stopped_) {
        throw CudaError("EventTimer: start() and stop() must be called before elapsed_ms()",
                        __FILE__, __LINE__);
    }
    float ms = 0.0f;
    LIBACCINT_CUDA_CHECK(cudaEventElapsedTime(&ms, start_event_.get(), stop_event_.get()));
    return ms;
}

// ============================================================================
// StreamPool Implementation
// ============================================================================

StreamPool::StreamPool(size_t n_streams, unsigned int flags) {
    streams_.reserve(n_streams);
    for (size_t i = 0; i < n_streams; ++i) {
        streams_.push_back(std::make_unique<StreamHandle>(flags));
        available_.push(streams_.back().get());
    }
}

StreamHandle& StreamPool::acquire() {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [this] { return !available_.empty(); });

    StreamHandle* stream = available_.front();
    available_.pop();
    return *stream;
}

void StreamPool::release(StreamHandle& stream) {
    std::lock_guard<std::mutex> lock(mutex_);
    available_.push(&stream);
    cv_.notify_one();
}

void StreamPool::synchronize_all() {
    // Note: We synchronize all streams, not just available ones,
    // because acquired streams may have pending work
    for (auto& stream : streams_) {
        stream->synchronize();
    }
}

size_t StreamPool::available() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return available_.size();
}

// ============================================================================
// ScopedStream Implementation
// ============================================================================

ScopedStream::ScopedStream(StreamPool& pool)
    : pool_(pool), stream_(&pool.acquire()) {
}

ScopedStream::~ScopedStream() {
    pool_.release(*stream_);
}

}  // namespace libaccint::memory

#endif  // LIBACCINT_USE_CUDA
