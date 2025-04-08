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
        __import__(package_name)
        return True
    except ImportError:
        return False

def install_package(package):
    """Install a package using pip"""
    console.print(f"[yellow]Installing {package}...[/yellow]")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        return True
    except subprocess.CalledProcessError:
        console.print(f"[red]Failed to install {package}[/red]")
        return False

def main():
    console.print(Panel.fit(
        "[bold cyan]Stable-TS Installation for VideoLingo[/bold cyan]\n\n"
        "This script will install stable-ts and its dependencies.",
        title="Installation"
    ))

    # Check if stable_whisper is already installed
    if check_package_installed("stable_whisper"):
        console.print("[green]✓ stable-whisper is already installed![/green]")
    else:
        console.print("[yellow]stable-whisper is not installed. Installing now...[/yellow]")

        # Install stable-whisper from the local directory
        stable_ts_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stable-ts")
        if os.path.exists(stable_ts_dir):
            console.print(f"[cyan]Installing stable-whisper from local directory: {stable_ts_dir}[/cyan]")
            try:
                # Install with editable mode to use the local version
                subprocess.check_call([sys.executable, "-m", "pip", "install", "-e", stable_ts_dir])
                console.print("[green]✓ Successfully installed stable-whisper from local directory![/green]")
            except subprocess.CalledProcessError:
                console.print("[red]Failed to install stable-whisper from local directory[/red]")
                console.print("[yellow]Trying to install from PyPI...[/yellow]")
                # Try to install the latest version from PyPI
                if not install_package("stable-whisper"):
                    console.print("[red]Failed to install stable-whisper from PyPI.[/red]")
                    console.print("[yellow]Trying to install a specific version...[/yellow]")
                    # Try a specific version that is known to work
                    if not install_package("stable-whisper==2.2.2"):
                        console.print("[red]Failed to install stable-whisper. Please install it manually.[/red]")
                        sys.exit(1)
                    else:
                        console.print("[green]✓ Successfully installed stable-whisper v2.2.2![/green]")
        else:
            console.print("[yellow]Local stable-ts directory not found. Installing from PyPI...[/yellow]")
            # Try to install the latest version from PyPI
            if not install_package("stable-whisper"):
                console.print("[red]Failed to install stable-whisper from PyPI.[/red]")
                console.print("[yellow]Trying to install a specific version...[/yellow]")
                # Try a specific version that is known to work
                if not install_package("stable-whisper==2.2.2"):
                    console.print("[red]Failed to install stable-whisper. Please install it manually.[/red]")
                    sys.exit(1)
                else:
                    console.print("[green]✓ Successfully installed stable-whisper v2.2.2![/green]")

    # Check and install other dependencies
    dependencies = [
        "torch",
        "librosa",
        "rich",
        "numpy",
        "ffmpeg-python"
    ]

    for dep in dependencies:
        if check_package_installed(dep.split("==")[0]):
            console.print(f"[green]✓ {dep} is already installed![/green]")
        else:
            if not install_package(dep):
                console.print(f"[red]Failed to install {dep}. Please install it manually.[/red]")

    # Check for Apple Silicon and install MLX if needed
    if sys.platform == "darwin" and "arm" in os.uname().machine:
        console.print("[cyan]Detected Apple Silicon. Checking for MLX support...[/cyan]")
        if check_package_installed("mlx"):
            console.print("[green]✓ MLX is already installed![/green]")
        else:
            console.print("[yellow]MLX is not installed. Installing now for Apple Silicon acceleration...[/yellow]")
            if not install_package("mlx"):
                console.print("[yellow]Failed to install MLX. Stable-TS will still work but without Apple Silicon acceleration.[/yellow]")

        if check_package_installed("mlx_whisper"):
            console.print("[green]✓ MLX-Whisper is already installed![/green]")
        else:
            console.print("[yellow]MLX-Whisper is not installed. Installing now for Apple Silicon acceleration...[/yellow]")
            if not install_package("mlx-whisper"):
                console.print("[yellow]Failed to install MLX-Whisper. Stable-TS will still work but without Apple Silicon acceleration.[/yellow]")

    console.print(Panel.fit(
        "[bold green]Installation Complete![/bold green]\n\n"
        "You can now use the stable-ts option in VideoLingo.\n"
        "To enable it, go to the settings and select 'stable-ts' as the WhisperX Runtime.",
        title="Success"
    ))

if __name__ == "__main__":
    main()
