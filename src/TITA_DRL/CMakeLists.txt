cmake_minimum_required(VERSION 3.8)
project(tita_drl)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(message_filters REQUIRED)



# 手动指定 TensorRT 的包含目录
include_directories(/usr/include/aarch64-linux-gnu/) # 确保该路径下包含 `NvInfer.h`
include_directories(/usr/include/eigen3)
include_directories(/usr/local/cuda-12.2/targets/aarch64-linux/include)

# 手动指定库目录
link_directories(/usr/lib/aarch64-linux-gnu/) # 确保该路径下包含 TensorRT 库
link_directories(/usr/local/cuda-12.2/lib64)
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

# 添加可执行文件
add_executable(tita_drl_obs57 src/tita_drl_obs57.cpp)
add_executable(text src/text.cpp)

# 链接库
target_link_libraries(tita_drl_obs57
    nvinfer # TensorRT 的核心库
    nvparsers # 用于解析 ONNX 模型的库（如果使用了）
    nvinfer_plugin # TensorRT 插件库（如果使用了插件）
    cudart # CUDA Runtime 库
)

ament_target_dependencies(tita_drl_obs57 rclcpp std_msgs sensor_msgs)

# 安装目标文件
install(TARGETS
  tita_drl_obs57
  text
  DESTINATION lib/${PROJECT_NAME})

# 安装 launch 和 config 目录
install(DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
