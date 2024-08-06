#pragma once

#include "cuda_utils.hpp"

namespace tiledcuda::cell::compute {

template <typename Element>
struct Add {
    DEVICE Element operator()(Element a, Element b) const { return a + b; }
};

template <typename Element>
struct Sub {
    DEVICE Element operator()(Element a, Element b) const { return a - b; }
};

template <typename Element>
struct Mul {
    DEVICE Element operator()(Element a, Element b) const { return a * b; }
};

template <typename Element>
struct Max {
    DEVICE Element operator()(Element a, Element b) const {
        return a > b ? a : b;
    }
};

template <typename Element>
struct Min {
    DEVICE Element operator()(Element a, Element b) const {
        return a < b ? a : b;
    }
};

template <typename Element>
struct Exp {
    DEVICE Element operator()(Element a) const { return exp(a); }
};

#if defined(__CUDA_ARCH__)
template <>
struct Exp<float> {
    DEVICE float operator()(float a) const { return __expf(a); }
};

template <>
struct Exp<__half> {
    DEVICE __half operator()(__half a) const { return hexp(a); }
};
#endif

template <typename Element>
struct Log {
    DEVICE Element operator()(Element a) const { return log(a); }
};

#if defined(__CUDA_ARCH__)
template <>
struct Log<float> {
    DEVICE float operator()(float a) const { return __logf(a); }
};

template <>
struct Log<__half> {
    DEVICE __half operator()(__half a) const { return hlog(a); }
};
#endif

}  // namespace tiledcuda::cell::compute
