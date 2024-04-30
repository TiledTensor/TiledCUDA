#include "common/test_utils.hpp"

int main(int argc, char** argv) {
    FLAGS_alsologtostderr = 1;  // redirect log to stderr
    google::InitGoogleLogging(argv[0]);

    testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}
