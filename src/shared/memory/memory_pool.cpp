// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Bryce M. Westheimer. All rights reserved.

/// @file memory_pool.cpp
/// @brief Memory pool implementation for efficient scratch buffer allocation

#include <libaccint/memory/memory_pool.hpp>
#include <cassert>

namespace libaccint::memory {

// ============================================================================
// PooledBuffer Implementation
// ============================================================================

PooledBuffer::~PooledBuffer() {
    if (ptr_ && pool_) {
        pool_->release(ptr_, size_class_);
    } else if (ptr_) {
        // Non-pooled allocation (oversized)
        aligned_free(ptr_);
    }
}

PooledBuffer& PooledBuffer::operator=(PooledBuffer&& other) noexcept {
    if (this != &other) {
        // Release current buffer
        if (ptr_ && pool_) {
            pool_->release(ptr_, size_class_);
        } else if (ptr_) {
            aligned_free(ptr_);
        }

        // Take ownership of other's buffer
        ptr_ = other.ptr_;
        size_ = other.size_;
        pool_ = other.pool_;
        size_class_ = other.size_class_;

        other.ptr_ = nullptr;
        other.size_ = 0;
        other.pool_ = nullptr;
        other.size_class_ = 0;
    }
    return *this;
}

// ============================================================================
// MemoryPool Implementation
// ============================================================================

PooledBuffer MemoryPool::acquire(std::size_t bytes) {
    if (bytes == 0) {
        return PooledBuffer{};
    }

    ++total_allocations_;

    // Find appropriate size class
    std::size_t size_class = pool_config::find_size_class(bytes);

    // Handle oversized allocations
    if (size_class >= pool_config::NUM_SIZE_CLASSES) {
        ++oversized_allocations_;
        void* ptr = aligned_malloc(bytes, DEFAULT_ALIGNMENT);
        if (!ptr) {
            throw std::bad_alloc();
        }
        // Return with null pool to indicate non-pooled
        return PooledBuffer{ptr, bytes, nullptr, pool_config::NUM_SIZE_CLASSES};
    }

    // Initialize free list size if needed
    if (free_lists_[size_class].buffer_size == 0) {
        free_lists_[size_class].buffer_size = pool_config::SIZE_CLASSES[size_class];
    }

    // Try to reuse from free list
    auto& free_list = free_lists_[size_class];
    if (!free_list.buffers.empty()) {
        ++pool_hits_;
        void* ptr = free_list.buffers.back();
        free_list.buffers.pop_back();
        return PooledBuffer{ptr, free_list.buffer_size, this, size_class};
    }

    // Allocate new buffer
    ++pool_misses_;
    void* ptr = aligned_malloc(free_list.buffer_size, DEFAULT_ALIGNMENT);
    if (!ptr) {
        throw std::bad_alloc();
    }

    return PooledBuffer{ptr, free_list.buffer_size, this, size_class};
}

void MemoryPool::release(void* ptr, std::size_t size_class) noexcept {
    if (!ptr) {
        return;
    }

    // Oversized allocations are freed directly
    if (size_class >= pool_config::NUM_SIZE_CLASSES) {
        aligned_free(ptr);
        return;
    }

    auto& free_list = free_lists_[size_class];

    // If free list is full, free the buffer
    if (free_list.buffers.size() >= pool_config::MAX_BUFFERS_PER_CLASS) {
        aligned_free(ptr);
        return;
    }

    // Return to free list
    free_list.buffers.push_back(ptr);

    // Track high-water mark
    std::size_t current_bytes = 0;
    for (std::size_t i = 0; i < pool_config::NUM_SIZE_CLASSES; ++i) {
        current_bytes += free_lists_[i].buffers.size() * free_lists_[i].buffer_size;
    }
    if (current_bytes > peak_pooled_bytes_) {
        peak_pooled_bytes_ = current_bytes;
    }
}

void MemoryPool::clear() noexcept {
    for (auto& free_list : free_lists_) {
        for (void* ptr : free_list.buffers) {
            aligned_free(ptr);
        }
        free_list.buffers.clear();
    }
}

void MemoryPool::trim(std::size_t max_per_class) noexcept {
    for (auto& free_list : free_lists_) {
        while (free_list.buffers.size() > max_per_class) {
            aligned_free(free_list.buffers.back());
            free_list.buffers.pop_back();
        }
    }
}

MemoryPool::Stats MemoryPool::stats() const noexcept {
    Stats s;
    s.total_allocations = total_allocations_;
    s.pool_hits = pool_hits_;
    s.pool_misses = pool_misses_;
    s.oversized_allocations = oversized_allocations_;

    for (std::size_t i = 0; i < pool_config::NUM_SIZE_CLASSES; ++i) {
        s.current_pooled += free_lists_[i].buffers.size();
        s.current_pooled_bytes += free_lists_[i].buffers.size() * free_lists_[i].buffer_size;
    }
    s.peak_pooled_bytes = peak_pooled_bytes_;

    return s;
}

// ============================================================================
// Thread-Local Pool
// ============================================================================

MemoryPool& get_thread_local_pool() noexcept {
    thread_local MemoryPool pool;
    return pool;
}

// ============================================================================
// Global Pool Implementation
// ============================================================================

GlobalMemoryPool& GlobalMemoryPool::instance() {
    static GlobalMemoryPool instance;
    return instance;
}

GlobalMemoryPool::~GlobalMemoryPool() {
    // Base class MemoryPool destructor handles clear().
    // No need to lock during shutdown — program is terminating.
}

PooledBuffer GlobalMemoryPool::acquire(std::size_t bytes) {
    std::lock_guard<std::mutex> lock(mutex_);
    // Calls MemoryPool::acquire() which creates PooledBuffer with 'this'
    // (the GlobalMemoryPool) as the pool back-pointer. Since release()
    // is virtual, PooledBuffer destructors will dispatch to our override.
    return MemoryPool::acquire(bytes);
}

void GlobalMemoryPool::release(void* ptr, std::size_t size_class) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    MemoryPool::release(ptr, size_class);
}

void GlobalMemoryPool::clear() noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    MemoryPool::clear();
}

MemoryPool::Stats GlobalMemoryPool::stats() const noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    return MemoryPool::stats();
}

}  // namespace libaccint::memory
