cmake_minimum_required(VERSION 3.25)
project(jvmwriter)

add_executable(jvmwriter main.cpp)
target_compile_options(jvmwriter PUBLIC -std=c++20 -Wall -Wextra -Wpedantic -fsanitize=address -fno-omit-frame-pointer -g3 -glldb -Og)
target_link_options(jvmwriter PUBLIC -std=c++20 -Wall -Wextra -Wpedantic -fsanitize=address -fno-omit-frame-pointer -g3 -glldb -Og)