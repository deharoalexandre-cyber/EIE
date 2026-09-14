#!/usr/bin/env python3
# EIE: assemble a distributable Windows x64 CPU bundle from a build directory.
# Usage: python scripts\bundle-windows.py [build-dir]   (default: build-windows-cpu)
# Output: dist\eie-windows-x64-<rev>\ and dist\eie-windows-x64-<rev>.zip (SHA-256 printed).
import hashlib, os, re, shutil, subprocess, sys, zipfile

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(root)
build = sys.argv[1] if len(sys.argv) > 1 else "build-windows-cpu"
exe = next((p for p in (os.path.join(build, "eie-server.exe"), os.path.join(build, "Release", "eie-server.exe")) if os.path.isfile(p)), None)
if not exe: sys.exit(f"{build}\eie-server.exe missing (see scripts\build-windows-cpu.bat)")
rev = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
name = f"eie-windows-x64-{rev}"
out = os.path.join("dist", name)
shutil.rmtree(out, ignore_errors=True)
os.makedirs(os.path.join(out, "presets"))
os.makedirs(os.path.join(out, "models"))
shutil.copy(exe, os.path.join(out, "eie-server.exe"))
for p in ("windows-cpu.yaml", "generic.yaml"): shutil.copy(os.path.join("presets", p), os.path.join(out, "presets", p))
shutil.copy(os.path.join("scripts", "start-eie-windows.bat"), os.path.join(out, "start-eie.bat"))
shutil.copy(os.path.join("docs", "windows.md"), os.path.join(out, "README.md"))
for p in ("LICENSE", "NOTICE"):
    if os.path.isfile(p): shutil.copy(p, out)
open(os.path.join(out, "models", "put-gguf-files-here.txt"), "w").write("Drop your GGUF files here (chat model, plus an embedding model whose name contains 'bge' or 'embed').\n")

# Toolchain record read by scripts/receipt-windows.py
cache = open(os.path.join(build, "CMakeCache.txt"), encoding="utf-8", errors="replace").read()
def cval(k):
    m = re.search(rf"^{k}:\w+=(.*)$", cache, re.M); return m.group(1).strip() if m else "?"
cxx = cval("CMAKE_CXX_COMPILER")
try: cxx_ver = subprocess.check_output([cxx, "--version"], text=True, stderr=subprocess.STDOUT).splitlines()[0]
except Exception: cxx_ver = os.path.basename(cxx)
info = (f"compiler: {cxx_ver}\ngenerator: {cval('CMAKE_GENERATOR')}\nbuild_type: {cval('CMAKE_BUILD_TYPE')}\n"
        f"GGML_NATIVE: {cval('GGML_NATIVE')}\nBUILD_SHARED_LIBS: {cval('BUILD_SHARED_LIBS')}\n"
        f"linker_flags: {cval('CMAKE_EXE_LINKER_FLAGS')}\nEIE: {rev}\n")
open(os.path.join(out, "build-info.txt"), "w", encoding="utf-8").write(info)

def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""): h.update(c)
    return h.hexdigest()
files = ["eie-server.exe", "start-eie.bat", "build-info.txt"] + [f"presets/{p}" for p in sorted(os.listdir(os.path.join(out, "presets")))]
with open(os.path.join(out, "SHA256SUMS"), "w", newline="\n") as f:
    for p in files: f.write(f"{sha(os.path.join(out, p))} *{p}\n")

zpath = os.path.join("dist", name + ".zip")
with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
    for d, _, fs in os.walk(out):
        for fn in fs:
            full = os.path.join(d, fn); z.write(full, os.path.relpath(full, "dist"))
print(info)
print(f"{sha(zpath)}  {zpath}")
