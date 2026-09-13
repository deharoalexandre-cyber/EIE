"""Build and package EIE for Android arm64 with an explicit CPU instruction set."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]
ARCH = {"dotprod": "armv8.2-a+dotprod+fp16", "i8mm": "armv8.6-a+dotprod+i8mm+nosve"}


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def capture(args, cwd=ROOT):
    return subprocess.check_output(args, cwd=cwd).decode("utf-8").strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ndk", type=Path, required=True)
    parser.add_argument("--variant", choices=ARCH, required=True)
    parser.add_argument("--jobs", type=int, default=6)
    parser.add_argument("--output-dir", type=Path, help="New package destination; existing bundles are never overwritten")
    args = parser.parse_args()
    ndk = args.ndk.resolve(strict=True)
    runtime = ROOT / "llama.cpp"
    if not (runtime / "src/llama.cpp").is_file():
        raise SystemExit("Initialize the pinned llama.cpp submodule and apply the EIE runtime patch first.")
    if "ews_n_slots" not in (runtime / "include/llama.h").read_text():
        raise SystemExit("The published EIE runtime patch is required, including for non-EWS builds.")
    hosts = list((ndk / "toolchains/llvm/prebuilt").iterdir())
    host = next(p for p in hosts if p.is_dir() and (p / "bin").is_dir())
    suffix = ".exe" if os.name == "nt" else ""
    readelf = host / "bin" / ("llvm-readelf" + suffix)
    build = ROOT / ("build-android-" + args.variant)
    build.mkdir(exist_ok=True)
    commands = []

    def run(command, filename):
        commands.append(command)
        with (build / filename).open("wb") as log:
            process = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        print(f"{filename}: exit {process.returncode}", flush=True)
        if process.returncode:
            print((build / filename).read_text(errors="replace")[-6000:])
            raise SystemExit(process.returncode)

    run(["cmake", "-S", str(ROOT), "-B", str(build), "-G", "Ninja",
         "-U", "HAVE_*", "-DCMAKE_TOOLCHAIN_FILE=" + str(ndk / "build/cmake/android.toolchain.cmake"),
         "-DANDROID_ABI=arm64-v8a", "-DANDROID_PLATFORM=android-26", "-DANDROID_STL=c++_shared",
         "-DCMAKE_BUILD_TYPE=Release", "-DBUILD_SHARED_LIBS=ON", "-DGGML_NATIVE=OFF",
         "-DGGML_CPU_ARM_ARCH=" + ARCH[args.variant], "-DGGML_CPU_KLEIDIAI=OFF",
         "-DGGML_LLAMAFILE=OFF", "-DGGML_OPENMP=OFF", "-DGGML_CUDA=OFF", "-DGGML_HIP=OFF",
         "-DGGML_OPENCL=OFF", "-DGGML_HEXAGON=OFF", "-DGGML_METAL=OFF",
         "-DLLAMA_OPENSSL=OFF", "-DLLAMA_BUILD_MTMD=OFF", "-DEIE_BUILD_SERVING_TESTS=ON"],
        "configure.log")
    run(["cmake", "--build", str(build), "--target", "eie-server", "eie-serving-tests",
         "--parallel", str(args.jobs)], "build.log")

    name = "eie-android-arm64-" + args.variant
    output = args.output_dir.resolve() if args.output_dir else build
    bundle = output / "bundle" / name
    if bundle.exists():
        raise SystemExit(f"Bundle already exists: {bundle}; use a fresh build directory to avoid stale files.")
    for directory in ("bin", "lib", "include", "licenses", "evidence"):
        (bundle / directory).mkdir(parents=True, exist_ok=True)
    shutil.copy2(build / "eie-server", bundle / "bin/eie-server")
    shutil.copy2(build / "tests/serving/eie-serving-tests", bundle / "bin/eie-serving-tests")
    libraries = list((build / "bin").glob("*.so"))
    if not any(p.name == "libggml-cpu.so" for p in libraries):
        raise SystemExit("Missing CPU backend")
    for library in libraries:
        shutil.copy2(library, bundle / "lib" / library.name)
    shutil.copy2(host / "sysroot/usr/lib/aarch64-linux-android/libc++_shared.so", bundle / "lib/libc++_shared.so")
    for header in list((runtime / "ggml/include").glob("*.h")) + [runtime / "include/llama.h"]:
        shutil.copy2(header, bundle / "include" / header.name)
    for source, name_out in [(ROOT / "LICENSE", "EIE-Apache-2.0.txt"), (ROOT / "NOTICE", "EIE-NOTICE.txt"),
                             (runtime / "LICENSE", "llama.cpp-MIT.txt"),
                             (runtime / "vendor/cpp-httplib/LICENSE", "cpp-httplib-MIT.txt"),
                             (ROOT / "mobile/licenses/nlohmann-MIT.txt", "nlohmann-MIT.txt"),
                             (ndk / "NOTICE", "Android-NDK-NOTICE.txt"),
                             (ndk / "NOTICE.toolchain", "Android-NDK-toolchain-NOTICE.txt")]:
        shutil.copy2(source, bundle / "licenses" / name_out)
    shutil.copy2(ROOT / "mobile/ANDROID_BUNDLE.md", bundle / "README.md")
    shutil.copy2(ROOT / "scripts/validate_android_neon.py", bundle / "validate_android_neon.py")
    for log in ("configure.log", "build.log"):
        shutil.copy2(build / log, bundle / "evidence" / log)
    elf_report = b""
    for binary in sorted((bundle / "lib").glob("*.so")) + sorted((bundle / "bin").iterdir()):
        elf_report += ("\nFILE " + binary.relative_to(bundle).as_posix() + "\n").encode()
        elf_report += subprocess.check_output([str(readelf), "-h", "-d", "-n", str(binary)])
    (bundle / "evidence/elf.txt").write_bytes(elf_report)
    manifest = {"schema": "eie.android.bundle/v1", "variant": args.variant, "cpu_arch": ARCH[args.variant],
                "abi": "arm64-v8a", "android_api_min": 26, "openmp": False, "gpu_offload": False,
                "source_revision": capture(["git", "rev-parse", "HEAD"]),
                "source_dirty": bool(capture(["git", "status", "--porcelain", "--untracked-files=no", "--ignore-submodules=all"])),
                "runtime_revision": capture(["git", "-C", str(runtime), "rev-parse", "HEAD"]),
                "runtime_patch_sha256": sha(ROOT / "patches/ews-runtime-2168b0.patch"),
                "runtime_diff_sha256": hashlib.sha256(subprocess.check_output(["git", "-C", str(runtime), "diff", "--binary", "HEAD"])).hexdigest(),
                "build_recipe_sha256": sha(Path(__file__)),
                "ndk": (ndk / "source.properties").read_text(), "commands": commands,
                "device_validation": "not run by build script; see separate on-device receipt",
                "files": {p.relative_to(bundle).as_posix(): {"sha256": sha(p), "bytes": p.stat().st_size}
                          for p in sorted(bundle.rglob("*")) if p.is_file()}}
    (bundle / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    checksums = [f"{sha(p)}  {p.relative_to(bundle).as_posix()}" for p in sorted(bundle.rglob("*")) if p.is_file()]
    (bundle / "SHA256SUMS.txt").write_text("\n".join(checksums) + "\n", encoding="utf-8")
    archive = output / (bundle.name + ".zip")
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED) as zipped:
        for path in sorted(bundle.rglob("*")):
            if path.is_file():
                zipped.write(path, path.relative_to(bundle.parent).as_posix())
    print(json.dumps({"archive": str(archive), "sha256": sha(archive), "bytes": archive.stat().st_size}), flush=True)


if __name__ == "__main__":
    main()
