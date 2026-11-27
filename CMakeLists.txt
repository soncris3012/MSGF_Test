cmake_minimum_required(VERSION 3.20)
project(MSGF_Test VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(MSGF_BUILD_GUI "Build Qt 6 MsgfTirApp GUI and reusable MsgfForm widget" ON)
option(MSGF_BUILD_TESTS "Build unit tests" ON)

if(MSVC)
    add_compile_options(/W4 /O2)
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs)

# 1. Pure C++17 + OpenCV Core Library (no Qt dependency)
add_library(msgf_core STATIC
    core/MsgfParams.h
    core/MsgfAlgorithm.h
    core/MsgfAlgorithm.cpp
    core/MsgfMetrics.h
    core/MsgfMetrics.cpp
    core/MsgfBaselines.h
    core/MsgfBaselines.cpp
    core/N5Algorithm.h
    core/N5Algorithm.cpp
    core/DatasetManager.h
    core/DatasetManager.cpp
    core/BenchmarkTypes.h
    core/BenchmarkExport.h
)
target_include_directories(msgf_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(msgf_core PUBLIC ${OpenCV_LIBS})

# 2. Headless CLI Benchmark Tool
add_executable(msgf_cli
    cli/msgf_cli.cpp
)
target_link_libraries(msgf_cli PRIVATE msgf_core)

# 3. Reusable Qt 6 Widget Module + Standalone Application
if(MSGF_BUILD_GUI)
    find_package(Qt6 REQUIRED COMPONENTS Core Gui Widgets Concurrent Charts)

    add_library(msgf_widget STATIC
        ui/MsgfForm.ui
        ui/MsgfForm.h
        ui/MsgfForm.cpp
        ui/N5Form.ui
        ui/N5Form.h
        ui/N5Form.cpp
        ui/ImageCanvas.h
        ui/ImageCanvas.cpp
        ui/DatasetBenchmarkWidget.h
        ui/DatasetBenchmarkWidget.cpp
        ui/NumericTableItem.h
    )
    set_target_properties(msgf_widget PROPERTIES
        AUTOMOC ON
        AUTOUIC ON
        AUTORCC ON
    )
    target_include_directories(msgf_widget PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
    )
    target_link_libraries(msgf_widget PUBLIC
        msgf_core
        Qt6::Core
        Qt6::Gui
        Qt6::Widgets
        Qt6::Concurrent
        Qt6::Charts
    )

    add_executable(MsgfTirApp
        main.cpp
    )
    target_link_libraries(MsgfTirApp PRIVATE msgf_widget)
endif()

# 4. Automated Unit & Mathematical Invariant Tests
if(MSGF_BUILD_TESTS)
    enable_testing()
    add_executable(msgf_tests
        tests/test_msgf.cpp
    )
    target_link_libraries(msgf_tests PRIVATE msgf_core)
    add_test(NAME msgf_unit_tests COMMAND msgf_tests)

    if(MSGF_BUILD_GUI)
        find_package(Qt6 REQUIRED COMPONENTS Test)
        add_executable(msgf_ui_tests
            tests/test_ui_benchmark.cpp
        )
        target_link_libraries(msgf_ui_tests PRIVATE msgf_widget Qt6::Test)
        add_test(NAME msgf_ui_tests COMMAND msgf_ui_tests)
    endif()
endif()
