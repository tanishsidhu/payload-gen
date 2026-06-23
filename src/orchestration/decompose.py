"""Turn a vague user request into an ordered list of API operations."""

# Rule-based multi-step flows (v1 — shows the seam exists, not full planning).
MULTI_STEP_TRIGGERS = {
    "book a deal": ["create_asset", "book_lease"],
    "book the deal": ["create_asset", "book_lease"],
    "onboard a lease": ["create_asset", "book_lease"],
    "onboard new lease": ["create_asset", "book_lease"],
    "new lease deal": ["create_asset", "book_lease"],
}

# Single-operation keyword hints (aligned with retrieval layer).
OPERATION_KEYWORDS = {
    "create_asset": ["create asset", "register asset", "new asset", "add equipment"],
    "book_lease": ["book lease", "book a lease", "new lease", "lease booking"],
    "post_receipt": ["post receipt", "post a receipt", "record receipt", "cash received"],
    "amend_contract": ["amend contract", "amendment", "rate change", "modify contract"],
}


def _detect_single_operation(text):
    for operation, phrases in OPERATION_KEYWORDS.items():
        for phrase in phrases:
            if phrase in text:
                return operation
    return None


def decompose(request, hinted_operation=None):
    """
    Return an ordered list of operation names to execute.

    hinted_operation: best op from Layer A retrieval (used when request is specific).
    """
    text = request.lower()

    for phrase, operations in MULTI_STEP_TRIGGERS.items():
        if phrase in text:
            return list(operations)

    detected = _detect_single_operation(text)
    if detected:
        return [detected]

    if hinted_operation:
        return [hinted_operation]

    return ["book_lease"]
