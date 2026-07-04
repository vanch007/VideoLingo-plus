"""Core package for VideoLingo-plus.

Submodules are intentionally imported lazily by Python when requested. Eagerly
loading every pipeline step here makes `python -m core.<module>` emit runpy
warnings and can trigger optional dependency imports during unrelated commands.
"""
