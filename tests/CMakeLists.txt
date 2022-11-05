cmake_minimum_required(VERSION 3.14)

set(TESTFILES
    main.cpp
    neuron.cpp
    receptive_field.cpp
    spike_train.cpp
    learning.cpp
    utils.cpp
    network.cpp)

set(TEST_MAIN unit_tests) # default name for test executable
set(TEST_RUNNER_PARAMS "")

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable(${TEST_MAIN} ${TESTFILES})
target_link_libraries(${TEST_MAIN} PRIVATE ${LIBRARY_NAME} doctest)
target_link_libraries(${TEST_MAIN} PUBLIC ${OpenCV_LIBS})
set_target_properties(${TEST_MAIN} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})

set_target_properties(${TEST_MAIN} PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO
)


add_test(
    # Use some per-module/project prefix so that it is easier to run only tests for this module
    NAME ${LIBRARY_NAME}.${TEST_MAIN}
    COMMAND ${TEST_MAIN} ${TEST_RUNNER_PARAMS})

include(CodeCoverage)
