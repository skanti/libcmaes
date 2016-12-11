#pragma once

#include <stdlib.h>
#include <cstddef>
#include <stdexcept>
#include <new>

template<typename T, std::size_t Alignment>
class AlignedAllocator {
public:

    // The following will be the same for virtually all allocators.
    typedef T *pointer;
    typedef const T *const_pointer;
    typedef T &reference;
    typedef const T &const_reference;
    typedef T value_type;
    typedef std::size_t size_type;
    typedef ptrdiff_t difference_type;

    T *address(T &r) const {
        return &r;
    }

    const T *address(const T &s) const {
        return &s;
    }

    std::size_t max_size() const {
        return (static_cast<std::size_t>(0) - static_cast<std::size_t>(1)) / sizeof(T);
    }


    // The following must be the same for all allocators.
    template<typename U>
    struct rebind {
        typedef AlignedAllocator<U, Alignment> other;
    };

    bool operator!=(const AlignedAllocator &other) const {
        return !(*this == other);
    }

    void construct(T *const p, const T &t) const {
        void *const pv = static_cast<void *>(p);

        new(pv) T(t);
    }

    void destroy(T *const p) const {
        p->~T();
    }

    bool operator==(const AlignedAllocator &other) const {
        return true;
    }


    // Default constructor, copy constructor, rebinding constructor, and destructor.
    // Empty for stateless allocators.
    AlignedAllocator() {}

    AlignedAllocator(const AlignedAllocator &) {}

    template<typename U>
    AlignedAllocator(const AlignedAllocator<U, Alignment> &) {}

    ~AlignedAllocator() {}


    // The following will be different for each allocator.
    T *allocate(const std::size_t n) const {
        if (n == 0) {
            return NULL;
        }

        // All allocators should contain an integer overflow check.
        // The Standardization Committee recommends that std::length_error
        // be thrown in the case of integer overflow.
        if (n > max_size()) {
            throw std::length_error("AlignedAllocator<T>::allocate() - Integer overflow.");
        }

        // Mallocator wraps malloc().
        void *pv = nullptr;
        posix_memalign(&pv, Alignment, n * sizeof(T));

        // Allocators should throw std::bad_alloc in the case of memory allocation failure.
        if (pv == NULL) {
            throw std::bad_alloc();
        }

        return static_cast<T *>(pv);
    }

    void deallocate(T *const p, const std::size_t n) const {
        free(p);
    }


    // The following will be the same for all allocators that ignore hints.
    template<typename U>
    T *allocate(const std::size_t n, const U * /* const hint */) const {
        return allocate(n);
    }


private:
    AlignedAllocator &operator=(const AlignedAllocator &);
};
