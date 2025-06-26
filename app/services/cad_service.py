"""CAD-related service functions."""

from ..relay.vlm_client import call_vlm


def analyze(data):
    """Stub function to analyze CAD data."""
    # TODO: implement business logic
    response = call_vlm(data)
    return {"message": "analysis result", "vlm_response": response}
