@echo off
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvarsall.bat" x64
if errorlevel 1 (
    echo ERROR: vcvarsall.bat failed
    exit /b 1
)

echo Compiler: %VCToolsVersion%
echo Configuring CMake...
cmake -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo ^
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5 ^
    -DSWIFTSHADER_BUILD_TESTS=OFF ^
    -DSWIFTSHADER_BUILD_BENCHMARKS=OFF ^
    -S C:\swiftshaderCUDA ^
    -B C:\swiftshaderCUDA\out\build\x64-Release
if errorlevel 1 (
    echo ERROR: CMake configure failed
    exit /b 1
)

echo Building...
cmake --build C:\swiftshaderCUDA\out\build\x64-Release --target vk_swiftshader --clean-first -- -j%NUMBER_OF_PROCESSORS%
if errorlevel 1 (
    echo ERROR: Build failed
    exit /b 1
)

echo Build complete.
