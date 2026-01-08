"""
Speaker identification services for Atlas Brain.

Provides voice-based speaker recognition using embedding models
like Resemblyzer for enrolling and identifying known speakers.
"""

from .resemblyzer import ResemblyzerSpeakerID

__all__ = ["ResemblyzerSpeakerID"]
