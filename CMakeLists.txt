cmake_minimum_required(VERSION 3.8)
project(arm_bringup)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)

# 1. Install directories to share (We added rl_agent here)
install(DIRECTORY
  launch
  config
  meshes
  urdf
  rl_agent
  pick_place      # <--- ADDED THIS LINE
  DESTINATION share/${PROJECT_NAME}
)

# 2. Install executable node scripts to lib
# (We changed this from rl_agent/dep_any.py to just dep_any.py because it copies to share now too)
install(PROGRAMS
  rl_agent/dep_any.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()