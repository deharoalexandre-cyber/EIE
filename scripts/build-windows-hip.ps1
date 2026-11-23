# Build an isolated EIE/PrismML HIP binary on Windows for gfx1151 qualification.
# Run from an x64 Developer PowerShell after installing MSVC, Windows SDK,
# CMake and Ninja. The build directory is independent of any live EIE service.
param(
    [string]$BuildDirectory = 'build-windows-hip',
    [string]$GpuTarget = 'gfx1151'
)

$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$hipPath = $env:HIP_PATH
if (-not $hipPath -or -not (Test-Path -LiteralPath $hipPath -PathType Container)) {
    throw 'HIP_PATH must point to an installed AMD HIP SDK.'
}
$clang = Join-Path $hipPath 'bin\clang.exe'
$clangxx = Join-Path $hipPath 'bin\clang++.exe'
foreach ($path in @($clang, $clangxx, (Join-Path $repo 'llama-prism\src'))) {
    if (-not (Test-Path -LiteralPath $path)) { throw "Missing build input: $path" }
}
foreach ($tool in @('cmake', 'ninja', 'git')) {
    if (-not (Get-Command $tool -ErrorAction SilentlyContinue)) { throw "Missing tool: $tool" }
}
$env:CMAKE_PREFIX_PATH = $hipPath
$build = Join-Path $repo $BuildDirectory
Write-Host "EIE source: $repo"
Write-Host "PrismML revision: $(& git -C (Join-Path $repo 'llama-prism') rev-parse HEAD)"
Write-Host "HIP SDK: $hipPath"
Write-Host "GPU target: $GpuTarget"

$configure = @(
    '-S', $repo, '-B', $build, '-G', 'Ninja',
    "-DCMAKE_C_COMPILER=$clang",
    "-DCMAKE_CXX_COMPILER=$clangxx",
    '-DCMAKE_BUILD_TYPE=Release',
    '-DEIE_LLAMA_RUNTIME=prism',
    '-DGGML_HIP=ON',
    "-DGPU_TARGETS=$GpuTarget",
    '-DGGML_BACKEND_DL=ON',
    '-DGGML_NATIVE=OFF',
    '-DGGML_OPENMP=OFF',
    '-DGGML_HIP_ROCWMMA_FATTN=OFF',
    '-DLLAMA_OPENSSL=OFF',
    '-DLLAMA_CURL=OFF',
    '-DLLAMA_BUILD_MTMD=OFF'
)
& cmake @configure
if ($LASTEXITCODE -ne 0) { throw "CMake configure failed: $LASTEXITCODE" }
& cmake --build $build --target eie-server --parallel 8
if ($LASTEXITCODE -ne 0) { throw "EIE build failed: $LASTEXITCODE" }

$exe = Join-Path $build 'eie-server.exe'
$dllDir = Join-Path $build 'bin'
if (-not (Test-Path -LiteralPath $exe -PathType Leaf)) { throw "Missing built executable: $exe" }
if (-not (Test-Path -LiteralPath $dllDir -PathType Container)) { throw "Missing DLL directory: $dllDir" }
$bundleExe = Join-Path $dllDir 'eie-server.exe'
Copy-Item -LiteralPath $exe -Destination $bundleExe -Force
Write-Host "EIE executable: $exe"
Write-Host "Runnable copy beside DLLs: $bundleExe"
Get-FileHash -Algorithm SHA256 -LiteralPath $exe | Format-List Path,Hash
