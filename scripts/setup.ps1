# One-time setup on a fresh Windows 11 machine (idempotent: safe to re-run).
# Run from the repo root in PowerShell:  powershell -ExecutionPolicy Bypass -File scripts\setup.ps1
# Accept the UAC prompts winget shows.

$ErrorActionPreference = "Stop"

function Install-IfMissing($command, $id, $extra = @()) {
    if (Get-Command $command -ErrorAction SilentlyContinue) {
        Write-Host "ok: $command"
    } else {
        Write-Host "installing $id ..."
        winget install -e --id $id --accept-source-agreements --accept-package-agreements --silent @extra
    }
}

Install-IfMissing git Git.Git
Install-IfMissing gh GitHub.cli
Install-IfMissing rustup Rustlang.Rustup
# The Store's python.exe stub is 0 bytes; check for a real interpreter.
$py = Get-Command python -ErrorAction SilentlyContinue
if (-not $py -or (Get-Item $py.Source).Length -eq 0) {
    winget install -e --id Python.Python.3.12 --accept-source-agreements --accept-package-agreements --silent
}
# MSVC linker for host-side cargo test / build scripts.
if (-not (Test-Path "${env:ProgramFiles(x86)}\Microsoft Visual Studio\2022\BuildTools")) {
    winget install -e --id Microsoft.VisualStudio.2022.BuildTools --accept-source-agreements --accept-package-agreements `
        --override "--quiet --wait --norestart --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"
}

# Pick up PATH changes from the installers.
$env:Path = [Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [Environment]::GetEnvironmentVariable("Path", "User") + ";$env:USERPROFILE\.cargo\bin"

# Rust toolchain pinned by rust-toolchain.toml (1.99.0 + wasm32-wasip1 + rustfmt/clippy).
rustup show | Out-Null

if (-not (Test-Path .venv)) { python -m venv .venv }
.\.venv\Scripts\python.exe -m pip install --upgrade pip --quiet
.\.venv\Scripts\python.exe -m pip install --upgrade "cambc>=3.0.1"
.\.venv\Scripts\python.exe -m pip install -e ".[dev]" --quiet
.\.venv\Scripts\pre-commit.exe install --hook-type pre-commit --hook-type pre-push

Write-Host ""
Write-Host "Done. Activate the venv with  .\.venv\Scripts\Activate.ps1  then try:"
Write-Host "  bctools gate main            # all maps, both sides vs the starter bot"
Write-Host "  cargo test --workspace"
