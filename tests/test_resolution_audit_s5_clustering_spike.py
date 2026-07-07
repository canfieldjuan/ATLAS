"""S5 current-code clustering calibration for issue #1993.

The implementation slice replaces these current-behavior expectations with the
accepted corrected grouping behavior.
"""

from __future__ import annotations

from typing import Sequence

from extracted_content_pipeline.support_ticket_clustering import (
    assign_support_ticket_clusters_with_diagnostics,
)
from extracted_content_pipeline.support_ticket_input_package import (
    build_support_ticket_input_package,
)
from extracted_content_pipeline.ticket_faq_markdown import build_ticket_faq_markdown


_SSO_SAME_INTENT_ROWS = (
    "SSO shows invalid SAML response",
    "Okta returns audience mismatch during single sign on",
    "Azure AD says assertion signature is invalid",
    "Identity provider rejects the ACS URL",
    "OneLogin metadata certificate expired",
    "SAML login fails with recipient mismatch",
    "Enterprise login cannot complete after IdP update",
    "Single sign-on redirects back to error page",
    "SAML assertion not accepted by service provider",
    "SSO callback says issuer is wrong",
    "IdP certificate rotation broke company login",
    "Federated login error after metadata change",
)

_CANCEL_SUBSCRIPTION_ROWS = (
    "cancel my subscription",
    "how do I cancel my monthly plan",
    "stop my recurring billing",
    "end my membership",
    "turn off auto-renew",
    "i want to cancel the subscription please",
    "unsubscribe me from the paid plan",
    "cancel recurring payments",
    "close my subscription",
    "cancel paid account",
)

_CANCEL_ORDER_ROWS = (
    "cancel my order",
    "cancel order 12345",
    "i need to cancel a purchase",
    "stop my order from shipping",
    "cancel the item I just bought",
    "how do I cancel an order",
    "please cancel my recent order",
    "void my order",
    "cancel this purchase",
    "kill order 998",
)


class _ConstantEmbeddingPort:
    def __init__(self) -> None:
        self.calls: list[tuple[str, ...]] = []

    def embed_texts(self, texts: Sequence[str]) -> tuple[tuple[float, float], ...]:
        self.calls.append(tuple(texts))
        return tuple((1.0, 0.0) for _ in texts)


def _ticket_rows(prefix: str, texts: Sequence[str]) -> list[dict[str, str]]:
    return [
        {
            "ticket_id": f"{prefix}-{index}",
            "subject": text.title(),
            "description": text,
        }
        for index, text in enumerate(texts, start=1)
    ]


def test_s5_current_code_fragments_same_intent_sso_rows_below_repeat_gate() -> None:
    package = build_support_ticket_input_package(_ticket_rows("sso", _SSO_SAME_INTENT_ROWS))

    result = build_ticket_faq_markdown(package.inputs["source_material"], max_items=0)

    assert package.inputs["top_ticket_clusters"] == [
        {"label": "login", "count": 10},
        {"label": "acs identity provider reject", "count": 1},
        {"label": "certificate expired metadata onelogin", "count": 1},
    ]
    assert result.items == ()
    assert result.non_repeat_ticket_count == 12
    assert result.non_repeat_question_count == 12


def test_s5_current_embedding_booster_cannot_cross_hard_topic_partitions() -> None:
    package = build_support_ticket_input_package(_ticket_rows("sso", _SSO_SAME_INTENT_ROWS))
    port = _ConstantEmbeddingPort()
    semantic_merges: list[dict[str, object]] = []

    result = build_ticket_faq_markdown(
        package.inputs["source_material"],
        max_items=0,
        embedding_port=port,
        embedding_merge_recorder=semantic_merges.append,
    )

    assert [len(batch) for batch in port.calls] == [10]
    embedded_texts = set(port.calls[0])
    assert "Identity Provider Rejects The Acs Url" not in embedded_texts
    assert "Onelogin Metadata Certificate Expired" not in embedded_texts
    assert semantic_merges == []
    assert result.items == ()
    assert result.non_repeat_ticket_count == 12


def test_s5_current_code_overmerges_cancel_subscription_and_order_intents() -> None:
    package = build_support_ticket_input_package(
        [
            *_ticket_rows("sub", _CANCEL_SUBSCRIPTION_ROWS),
            *_ticket_rows("order", _CANCEL_ORDER_ROWS),
        ]
    )

    result = build_ticket_faq_markdown(package.inputs["source_material"], max_items=0)

    assert package.inputs["top_ticket_clusters"][0] == {"label": "order", "count": 17}
    assert len(result.items) == 1
    item = result.items[0]
    assert item["topic"] == "order"
    assert item["ticket_count"] == 11
    assert any(source_id.startswith("sub-") for source_id in item["source_ids"])
    assert any(source_id.startswith("order-") for source_id in item["source_ids"])
    assert result.non_repeat_ticket_count == 9


def test_s5_current_output_order_and_source_id_order_follow_input_order() -> None:
    rows = [
        {
            "source_type": "support_ticket",
            "support_ticket_cluster": "refunds",
            "source_id": "refund-a1",
            "source_title": "Refund A",
            "text": "How do I get my money back?",
        },
        {
            "source_type": "support_ticket",
            "support_ticket_cluster": "refunds",
            "source_id": "refund-a2",
            "source_title": "Refund B",
            "text": "How do I get my money back?",
        },
        {
            "source_type": "support_ticket",
            "support_ticket_cluster": "refunds",
            "source_id": "refund-b1",
            "source_title": "Credit A",
            "text": "Where can I request a refund credit?",
        },
        {
            "source_type": "support_ticket",
            "support_ticket_cluster": "refunds",
            "source_id": "refund-b2",
            "source_title": "Credit B",
            "text": "Where can I request a refund credit?",
        },
    ]

    forward = build_ticket_faq_markdown(rows, max_items=0)
    reverse = build_ticket_faq_markdown(list(reversed(rows)), max_items=0)

    assert [(item["question"], item["source_ids"]) for item in forward.items] == [
        ("How do I get my money back?", ("refund-a1", "refund-a2")),
        ("Where can I request a refund credit?", ("refund-b1", "refund-b2")),
    ]
    assert [(item["question"], item["source_ids"]) for item in reverse.items] == [
        ("Where can I request a refund credit?", ("refund-b2", "refund-b1")),
        ("How do I get my money back?", ("refund-a2", "refund-a1")),
    ]


def test_s5_current_token_set_skip_leaves_large_preview_rows_uncategorized() -> None:
    rows = [
        {
            "ticket_id": f"large-{index}",
            "description": (
                f"Widget depot transfer {index} misroutes label batch {index % 3}"
            ),
        }
        for index in range(4)
    ]

    annotated, diagnostics = assign_support_ticket_clusters_with_diagnostics(
        rows,
        max_token_set_rows=3,
    )

    assert diagnostics == {
        "token_set_row_count": 4,
        "max_token_set_rows": 3,
        "cluster_preview_skipped": True,
    }
    assert all("support_ticket_cluster" not in row for row in annotated)
