#include "common/test_utils.hpp"
#include "types/mod.hpp"

namespace tiledcuda::testing {

using namespace cell;
namespace tl = tile_layout;

TEST(TestLayout, test) {
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

}  // namespace tiledcuda::testing
