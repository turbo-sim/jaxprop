"""Setup/serialization identities for native source objects.

A CoolProp handle cannot be pickled. Recreating a handle from its fluid name
alone is also insufficient: global energy references, fluid definitions and
native backend files may have changed. Capture the same complete provenance
used for table generation once at setup, and compare it only when restoring a
serialized model. No identity probes belong in repeated state calculations.
"""

from ..errors import InvalidConfigurationError


def source_serialization_identity(name, backend, reference_id):
    """Return a verifiable source digest, or why serialization is unsupported.

    Reuse the existing source-signature implementation so serialized sources
    and generated tables agree about configuration, references and native-file
    identity. Some usable source backends cannot expose verifiable definitions;
    those remain usable for evaluation, but cannot promise a lossless restore.
    """
    from ..bicubic.cache_metadata import metadata_fingerprint, source_signature

    try:
        signature = source_signature(name, backend)
    except InvalidConfigurationError as error:
        return None, error.reason
    if signature["source_reference_id"] != reference_id:
        raise InvalidConfigurationError(
            fluid_name=name,
            provider="coolprop",
            model="real_fluid",
            request="source_reference",
            reason="The source convention changed while preparing serialization metadata.",
        )
    if not signature["deterministic_cache_identity"]:
        return None, "The backend does not expose verifiable native source definitions."
    return metadata_fingerprint(signature), None
