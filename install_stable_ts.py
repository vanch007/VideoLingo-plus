#!/usr/bin/env python3
import os
import sys
import subprocess
from rich.console import Console
from rich.panel import Panel

console = Console()

def check_package_installed(package_name):
    """Check if a package is installed"""
    try:
        # A more robust way to check for installation
        subprocess.check_call([sys.executable, "-m", "pip", "show", package_name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except subprocess.CalledProcessError:
        return False

def install_package(package, upgrade=False):
    """Install a package using pip"""
    cmd = [sys.executable, "-m", "pip", "install"]
    if upgrade:
        cmd.append("--upgrade")
    
    # Add git+ for github urls
    if "github.com" in package:
        cmd.append(package)
    else:
        cmd.append(package)

    console.print(f"[yellow]Running: {' '.join(cmd)}[/yellow]")
    try:
        subprocess.check_call(cmd)
        return True
    except subprocess.CalledProcessError:
        console.print(f"[red]Failed to install/upgrade {package}[/red]")
        return False

def main():
    console.print(Panel.fit(
        "[bold cyan]Stable-TS Installation for VideoLingo[/bold cyan]\n\n"
        "This script will install or upgrade stable-ts and its dependencies.\n"
        "- Installs/upgrades stable-whisper directly from GitHub for the latest version.\n"
        "- Installs required dependencies (torch, librosa, etc.).\n"
        "- Installs MLX support for Apple Silicon devices (if applicable).",
        title="Installation/Upgrade"
    ))

    # Always try to install/upgrade stable-whisper from GitHub
    console.print("[cyan]Attempting to install or upgrade stable-whisper from GitHub...[/cyan]")
    if not install_package("git+https://github.com/jianfch/stable-ts.git", upgrade=True):
        console.print("[red]Failed to install stable-whisper from GitHub.[/red]")
        console.print("[yellow]Please try to install it manually with:[/yellow]")
        console.print("[cyan]pip install --upgrade git+https://github.com/jianfch/stable-ts.git[/cyan]")
        sys.exit(1)
    else:
        console.print("[green]✓ Successfully installed/upgraded stable-whisper from GitHub![/green]")


    # Check and install other dependencies
    dependencies = [
        "torch",
        "librosa",
        "rich",
        "numpy",
        "ffmpeg-python"
    ]

    for dep in dependencies:
        if not check_package_installed(dep.split("==")[0]):
            if not install_package(dep):
                console.print(f"[red]Failed to install {dep}. Please install it manually.[/red]")
        else:
            console.print(f"[green]✓ {dep} is already installed.[/green]")


    # Check for Apple Silicon and install MLX if needed
    if sys.platform == "darwin" and "arm" in os.uname().machine:
        console.print("[cyan]Detected Apple Silicon. Checking for MLX support...[/cyan]")
        if not check_package_installed("mlx"):
            console.print("[yellow]MLX is not installed. Installing now for Apple Silicon acceleration...[/yellow]")
            if not install_package("mlx"):
                console.print("[yellow]Failed to install MLX. Stable-TS will still work but without Apple Silicon acceleration.[/yellow]")
        else:
            console.print("[green]✓ MLX is already installed![/green]")

    console.print(Panel.fit(
        "[bold green]Installation Complete![/bold green]\n\n"
        "You can now use the stable-ts option in VideoLingo.\n"
        "To enable it, go to the settings and select 'stable-ts' as the WhisperX Runtime.",
        title="Success"
    ))

if __name__ == "__main__":
    main()