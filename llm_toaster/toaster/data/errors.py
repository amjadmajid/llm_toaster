"""Contextual exceptions for the data pipeline.

Every failure mode in the manifest/shard/producer path raises one of these so callers
can react precisely (e.g. ``wait`` mode catches :class:`DataExhausted`) and users get an
actionable message instead of a bare ``KeyError``/``OSError``.
"""

from __future__ import annotations


class DataError(Exception):
    """Base class for all data-pipeline errors."""


class ManifestError(DataError):
    """Malformed, inconsistent, or unreadable manifest."""


class ManifestVersionError(ManifestError):
    """The manifest ``format_version`` is newer than this build understands."""


class ShardError(DataError):
    """A shard file is missing, truncated, or fails its recorded checksum."""


class ChecksumMismatchError(ShardError):
    """A shard's on-disk bytes do not match the checksum recorded in the manifest."""


class DataExhausted(DataError):
    """Raised by a batch source when no more unique data is available.

    Under ``sampling.exhaustion: stop`` the engine catches this, ends training at a clean
    optimizer-step boundary, and saves a final checkpoint. It is a control signal, not a bug.
    """

    def __init__(self, message: str = "pretraining data exhausted", *, pass_index: int = 0):
        super().__init__(message)
        self.pass_index = pass_index


class DataWaitTimeout(DataError):
    """``wait`` mode waited longer than ``materialization.wait_timeout_s`` for a new shard."""


class ProducerFailedError(DataError):
    """The background prefetch producer reported a failure (status file ``failed``)."""


class ConcurrentProducerError(DataError):
    """A second manifest writer was detected for a store that allows only one at a time."""


class ResumeIncompatibleError(DataError):
    """A checkpoint cannot be resumed against the current dataset/manifest.

    Raised when dataset identity, source revision, tokenizer/transform fingerprint, the
    current shard checksum, or the committed manifest prefix has changed since the checkpoint
    was written. Failing here is far safer than silently training on different data.
    """


class HFDependencyError(DataError):
    """The optional Hugging Face ``datasets`` dependency is missing or too old."""
