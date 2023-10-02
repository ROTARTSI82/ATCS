cmake_minimum_required(VERSION 3.25)
project(anygrad CXX)

add_executable(anygrad main.cpp)

target_include_directories(anygrad PUBLIC .)
target_compile_options(anygrad PUBLIC -std=c++20 -Wall -Wextra -Wpedantic -Ofast -flto -march=native -mtune=native) # -g3 -glldb -fsanitize=address -fno-omit-frame-pointer -Og)
target_link_options(anygrad PUBLIC -std=c++20 -Wall -Wextra -Wpedantic -Ofast -flto -march=native -mtune=native) # -g3 -glldb -fsanitize=address -fno-omit-frame-pointer -Og)

