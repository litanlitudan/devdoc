cmake_minimum_required(VERSION 3.20)
project(MLIRParser C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find LLVM and MLIR
find_package(MLIR REQUIRED CONFIG)
find_package(LLVM REQUIRED CONFIG)

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})

# Find nlohmann/json
find_package(nlohmann_json 3.2.0 REQUIRED)

# Define the executable
add_executable(mlir_parser mlir_parser.cpp)

# Link against MLIR libraries
target_link_libraries(mlir_parser PRIVATE
    MLIRAnalysis
    MLIRDialect
    MLIRParser
    MLIRPass
    MLIRTransforms
    MLIRTransformUtils
    MLIRIR
    MLIRSupport

    # Dialect libraries
    MLIRFuncDialect
    MLIRArithDialect
    MLIRShapeDialect
    MLIRSCFDialect
    MLIRTensorDialect
    MLIRLinalgDialect
    MLIRTosaDialect
    MLIRMathDialect
    MLIRMemRefDialect
    MLIRGPUDialect
    MLIRVectorDialect
    MLIRControlFlowDialect

    # JSON library
    nlohmann_json::nlohmann_json
)

# Install target
install(TARGETS mlir_parser DESTINATION bin)

# Print build configuration
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Install prefix: ${CMAKE_INSTALL_PREFIX}")
