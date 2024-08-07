#include "cell/copy/global_to_shared.hpp"
#include "common/test_utils.hpp"
#include "types/mod.hpp"

#include <cute/layout.hpp>
#include <thrust/host_vector.h>

namespace tiledcuda::testing {

using namespace cell;
using namespace cute;
namespace tl = tile_layout;

/*
TEST(TestLayout, test_layout) {
    using Element = cutlass::half_t;

    using Layout1 = tl::RowMajor<4, 7>;
    EXPECT_EQ(tl::num_rows<Layout1>, 4);
    EXPECT_EQ(tl::num_cols<Layout1>, 7);
    EXPECT_EQ(tl::get_numel<Layout1>, 28);
    EXPECT_EQ(tl::row_stride<Layout1>, 7);
    EXPECT_EQ(tl::col_stride<Layout1>, 1);

    tl::Layout type1 = tl::layout_type<Layout1>;
    EXPECT_EQ(type1, tl::Layout::kRowMajor);
    auto layout_name1 = layout_type_to_str(type1);
    EXPECT_EQ(layout_name1, "RowMajor");

    using Layout2 = tl::ColMajor<4, 7>;
    EXPECT_EQ(tl::num_rows<Layout2>, 4);
    EXPECT_EQ(tl::num_cols<Layout2>, 7);
    EXPECT_EQ(tl::get_numel<Layout2>, 28);
    EXPECT_EQ(tl::row_stride<Layout2>, 1);
    EXPECT_EQ(tl::col_stride<Layout2>, 4);

    tl::Layout type2 = tl::layout_type<Layout2>;
    EXPECT_EQ(type2, tl::Layout::kColMajor);
    auto layout_name2 = layout_type_to_str(type2);
    EXPECT_EQ(layout_name2, "ColMajor");
}
*/

TEST(TestLayout, test_swizzled_layout) {
    using Element = __half;

    const int kRows = 16;
    const int kCols = 32;

    thrust::host_vector<Element> data(kRows * kCols);
    for (int i = 0; i < data.size(); ++i) {
        data[i] = static_cast<Element>(i % 2048);
    }

    using RowMajor = tl::RowMajor<kRows, 16, kCols>;
    RowMajor layout1;

    // only siwizzle the first [16x16] half of the [kRows, kCols] matrix
    using LayoutAtom = cute::Layout<Shape<_16, _16>, Stride<Int<kCols>, _1>>;
    using Swizzled = tl::SwizzledRowMajor<16 /*16bits*/, LayoutAtom>;
    Swizzled layout2;

    Element* ptr = thrust::raw_pointer_cast(data.data());

    printf("\nnon-swizzled:\n");
    for (int i = 0; i < RowMajor::kRows; ++i) {
        for (int j = 0; j < RowMajor::kCols; ++j) {
            printf("%.0f, ", __half2float(ptr[layout1(i, j)]));
        }
        printf("\n");
    }

    printf("\nswizzled:\n");
    for (int i = 0; i < kRows; ++i) {
        for (int j = 0; j < 16; ++j) {
            printf("%.0f, ", __half2float(ptr[layout2(i, j)]));
        }
        printf("\n");
    }
}

}  // namespace tiledcuda::testing
