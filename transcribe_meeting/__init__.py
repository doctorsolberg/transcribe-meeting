"""Local meeting transcription pipeline."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("transcribe-meeting")
except PackageNotFoundError:  # running from a source checkout without an install
    __version__ = "1.0.0"
