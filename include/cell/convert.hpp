#pragma once
#include "cuda_utils.hpp"

#include <cute/tensor.hpp>
#include <cutlass/numeric_conversion.h>

namespace tiledcuda::cell {

template <typename To_type, typename Engine, typename Layout>
DEVICE auto convert_type(cute::Tensor<Engine, Layout> const& tensor) {
    using From_type = typename Engine::value_type;
    constexpr int numel = decltype(size(tensor))::value;
    cutlass::NumericArrayConverter<To_type, From_type, numel> convert_op;
    // HACK: this requires tensor to be "contiguous"
    auto frag =
        convert_op(*reinterpret_cast<const cutlass::Array<From_type, numel>*>(
            tensor.data()));

    return make_tensor(make_rmem_ptr<To_type>(&frag), tensor.layout());
}

template <typename Tensor>
struct IndexedTensor_ {
    DEVICE IndexedTensor_(Tensor& tensor) : tensor_(tensor) {}

    DEVICE const auto operator[](int idx) { return tensor_(_, _, idx); }

   private:
    Tensor& tensor_;
};

// Convert acc_layout from (MMA=4, MMA_M, MMA_N) to
// ((4, 2), MMA_M, MMA_N / 2) if using m16n8k16, or to (4, MMA_M, MMA_N) if
// using m16n8k8.
template <typename MMA, typename Tensor>
DEVICE auto convert_layout(const Tensor& acc) {
    auto acc_layout = acc.layout();

    using X = Underscore;
    static_assert(decltype(size<0>(acc_layout))::value == 4);
    static_assert(decltype(cute::rank(acc_layout))::value == 3);

    constexpr int mma_shape_K = cute::get<2>(typename MMA::Shape_MNK{});
    static_assert(mma_shape_K == 8 || mma_shape_K == 16);

    if constexpr (mma_shape_K == 8) {
        IndexedTensor_<decltype(acc)> indexed_tensor(acc);
        return indexed_tensor;
    } else {
        // (4, MMA_M, (2, MMA_N / 2)))
        auto l = cute::logical_divide(acc_layout, Shape<X, X, _2>{});
        auto new_layout = make_layout(make_layout(get<0>(l), get<2, 0>(l)),
                                      get<1>(l), get<2, 1>(l));
        auto new_tensor = make_tensor(acc.data(), new_layout);

        IndexedTensor_<decltype(new_tensor)> indexed_tensor(new_tensor);
        return indexed_tensor;
    }
};
}  // namespace tiledcuda::cell
