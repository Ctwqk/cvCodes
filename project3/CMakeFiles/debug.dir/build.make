# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/taiwei/code/cvCode/project3

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/taiwei/code/cvCode/project3

# Include any dependencies generated for this target.
include CMakeFiles/debug.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/debug.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/debug.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/debug.dir/flags.make

CMakeFiles/debug.dir/debug.cpp.o: CMakeFiles/debug.dir/flags.make
CMakeFiles/debug.dir/debug.cpp.o: debug.cpp
CMakeFiles/debug.dir/debug.cpp.o: CMakeFiles/debug.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/taiwei/code/cvCode/project3/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/debug.dir/debug.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/debug.dir/debug.cpp.o -MF CMakeFiles/debug.dir/debug.cpp.o.d -o CMakeFiles/debug.dir/debug.cpp.o -c /home/taiwei/code/cvCode/project3/debug.cpp

CMakeFiles/debug.dir/debug.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/debug.dir/debug.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/taiwei/code/cvCode/project3/debug.cpp > CMakeFiles/debug.dir/debug.cpp.i

CMakeFiles/debug.dir/debug.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/debug.dir/debug.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/taiwei/code/cvCode/project3/debug.cpp -o CMakeFiles/debug.dir/debug.cpp.s

CMakeFiles/debug.dir/region.cpp.o: CMakeFiles/debug.dir/flags.make
CMakeFiles/debug.dir/region.cpp.o: region.cpp
CMakeFiles/debug.dir/region.cpp.o: CMakeFiles/debug.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/taiwei/code/cvCode/project3/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/debug.dir/region.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/debug.dir/region.cpp.o -MF CMakeFiles/debug.dir/region.cpp.o.d -o CMakeFiles/debug.dir/region.cpp.o -c /home/taiwei/code/cvCode/project3/region.cpp

CMakeFiles/debug.dir/region.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/debug.dir/region.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/taiwei/code/cvCode/project3/region.cpp > CMakeFiles/debug.dir/region.cpp.i

CMakeFiles/debug.dir/region.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/debug.dir/region.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/taiwei/code/cvCode/project3/region.cpp -o CMakeFiles/debug.dir/region.cpp.s

CMakeFiles/debug.dir/denoise.cpp.o: CMakeFiles/debug.dir/flags.make
CMakeFiles/debug.dir/denoise.cpp.o: denoise.cpp
CMakeFiles/debug.dir/denoise.cpp.o: CMakeFiles/debug.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/taiwei/code/cvCode/project3/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/debug.dir/denoise.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/debug.dir/denoise.cpp.o -MF CMakeFiles/debug.dir/denoise.cpp.o.d -o CMakeFiles/debug.dir/denoise.cpp.o -c /home/taiwei/code/cvCode/project3/denoise.cpp

CMakeFiles/debug.dir/denoise.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/debug.dir/denoise.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/taiwei/code/cvCode/project3/denoise.cpp > CMakeFiles/debug.dir/denoise.cpp.i

CMakeFiles/debug.dir/denoise.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/debug.dir/denoise.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/taiwei/code/cvCode/project3/denoise.cpp -o CMakeFiles/debug.dir/denoise.cpp.s

CMakeFiles/debug.dir/features.cpp.o: CMakeFiles/debug.dir/flags.make
CMakeFiles/debug.dir/features.cpp.o: features.cpp
CMakeFiles/debug.dir/features.cpp.o: CMakeFiles/debug.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/taiwei/code/cvCode/project3/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object CMakeFiles/debug.dir/features.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/debug.dir/features.cpp.o -MF CMakeFiles/debug.dir/features.cpp.o.d -o CMakeFiles/debug.dir/features.cpp.o -c /home/taiwei/code/cvCode/project3/features.cpp

CMakeFiles/debug.dir/features.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/debug.dir/features.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/taiwei/code/cvCode/project3/features.cpp > CMakeFiles/debug.dir/features.cpp.i

CMakeFiles/debug.dir/features.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/debug.dir/features.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/taiwei/code/cvCode/project3/features.cpp -o CMakeFiles/debug.dir/features.cpp.s

CMakeFiles/debug.dir/csv_util.cpp.o: CMakeFiles/debug.dir/flags.make
CMakeFiles/debug.dir/csv_util.cpp.o: csv_util.cpp
CMakeFiles/debug.dir/csv_util.cpp.o: CMakeFiles/debug.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/taiwei/code/cvCode/project3/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object CMakeFiles/debug.dir/csv_util.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/debug.dir/csv_util.cpp.o -MF CMakeFiles/debug.dir/csv_util.cpp.o.d -o CMakeFiles/debug.dir/csv_util.cpp.o -c /home/taiwei/code/cvCode/project3/csv_util.cpp

CMakeFiles/debug.dir/csv_util.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/debug.dir/csv_util.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/taiwei/code/cvCode/project3/csv_util.cpp > CMakeFiles/debug.dir/csv_util.cpp.i

CMakeFiles/debug.dir/csv_util.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/debug.dir/csv_util.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/taiwei/code/cvCode/project3/csv_util.cpp -o CMakeFiles/debug.dir/csv_util.cpp.s

CMakeFiles/debug.dir/NetWorkThings/dnnEmbedding.cpp.o: CMakeFiles/debug.dir/flags.make
CMakeFiles/debug.dir/NetWorkThings/dnnEmbedding.cpp.o: NetWorkThings/dnnEmbedding.cpp
CMakeFiles/debug.dir/NetWorkThings/dnnEmbedding.cpp.o: CMakeFiles/debug.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/taiwei/code/cvCode/project3/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building CXX object CMakeFiles/debug.dir/NetWorkThings/dnnEmbedding.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/debug.dir/NetWorkThings/dnnEmbedding.cpp.o -MF CMakeFiles/debug.dir/NetWorkThings/dnnEmbedding.cpp.o.d -o CMakeFiles/debug.dir/NetWorkThings/dnnEmbedding.cpp.o -c /home/taiwei/code/cvCode/project3/NetWorkThings/dnnEmbedding.cpp

CMakeFiles/debug.dir/NetWorkThings/dnnEmbedding.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/debug.dir/NetWorkThings/dnnEmbedding.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/taiwei/code/cvCode/project3/NetWorkThings/dnnEmbedding.cpp > CMakeFiles/debug.dir/NetWorkThings/dnnEmbedding.cpp.i

CMakeFiles/debug.dir/NetWorkThings/dnnEmbedding.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/debug.dir/NetWorkThings/dnnEmbedding.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/taiwei/code/cvCode/project3/NetWorkThings/dnnEmbedding.cpp -o CMakeFiles/debug.dir/NetWorkThings/dnnEmbedding.cpp.s

# Object files for target debug
debug_OBJECTS = \
"CMakeFiles/debug.dir/debug.cpp.o" \
"CMakeFiles/debug.dir/region.cpp.o" \
"CMakeFiles/debug.dir/denoise.cpp.o" \
"CMakeFiles/debug.dir/features.cpp.o" \
"CMakeFiles/debug.dir/csv_util.cpp.o" \
"CMakeFiles/debug.dir/NetWorkThings/dnnEmbedding.cpp.o"

# External object files for target debug
debug_EXTERNAL_OBJECTS =

debug: CMakeFiles/debug.dir/debug.cpp.o
debug: CMakeFiles/debug.dir/region.cpp.o
debug: CMakeFiles/debug.dir/denoise.cpp.o
debug: CMakeFiles/debug.dir/features.cpp.o
debug: CMakeFiles/debug.dir/csv_util.cpp.o
debug: CMakeFiles/debug.dir/NetWorkThings/dnnEmbedding.cpp.o
debug: CMakeFiles/debug.dir/build.make
debug: /usr/local/lib/libopencv_gapi.so.4.9.0
debug: /usr/local/lib/libopencv_highgui.so.4.9.0
debug: /usr/local/lib/libopencv_ml.so.4.9.0
debug: /usr/local/lib/libopencv_objdetect.so.4.9.0
debug: /usr/local/lib/libopencv_photo.so.4.9.0
debug: /usr/local/lib/libopencv_stitching.so.4.9.0
debug: /usr/local/lib/libopencv_video.so.4.9.0
debug: /usr/local/lib/libopencv_videoio.so.4.9.0
debug: /usr/local/lib/libopencv_imgcodecs.so.4.9.0
debug: /usr/local/lib/libopencv_dnn.so.4.9.0
debug: /usr/local/lib/libopencv_calib3d.so.4.9.0
debug: /usr/local/lib/libopencv_features2d.so.4.9.0
debug: /usr/local/lib/libopencv_flann.so.4.9.0
debug: /usr/local/lib/libopencv_imgproc.so.4.9.0
debug: /usr/local/lib/libopencv_core.so.4.9.0
debug: CMakeFiles/debug.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/taiwei/code/cvCode/project3/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Linking CXX executable debug"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/debug.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/debug.dir/build: debug
.PHONY : CMakeFiles/debug.dir/build

CMakeFiles/debug.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/debug.dir/cmake_clean.cmake
.PHONY : CMakeFiles/debug.dir/clean

CMakeFiles/debug.dir/depend:
	cd /home/taiwei/code/cvCode/project3 && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/taiwei/code/cvCode/project3 /home/taiwei/code/cvCode/project3 /home/taiwei/code/cvCode/project3 /home/taiwei/code/cvCode/project3 /home/taiwei/code/cvCode/project3/CMakeFiles/debug.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/debug.dir/depend
