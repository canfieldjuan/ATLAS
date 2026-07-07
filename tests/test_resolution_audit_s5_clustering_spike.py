"""S5 clustering acceptance coverage for issue #1993."""

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


class _PairEmbeddingPort:
    def __init__(self) -> None:
        self.calls: list[tuple[str, ...]] = []

    def embed_texts(self, texts: Sequence[str]) -> tuple[tuple[float, float], ...]:
        self.calls.append(tuple(texts))
        return ((1.0, 0.0), (0.99, 0.01))


def _ticket_rows(prefix: str, texts: Sequence[str]) -> list[dict[str, str]]:
    return [
        {
            "ticket_id": f"{prefix}-{index}",
            "subject": text.title(),
            "description": text,
        }
        for index, text in enumerate(texts, start=1)
    ]


def test_s5_sso_rows_do_not_render_fabricated_question() -> None:
    package = build_support_ticket_input_package(_ticket_rows("sso", _SSO_SAME_INTENT_ROWS))

    result = build_ticket_faq_markdown(package.inputs["source_material"], max_items=0)

    assert package.inputs["top_ticket_clusters"] == [{"label": "login", "count": 12}]
    assert result.items == ()
    assert "How do I fix SSO login?" not in result.markdown
    assert result.non_repeat_ticket_count == 12
    assert result.non_repeat_question_count == 12


def test_s5_embedding_booster_can_cross_generated_token_partitions() -> None:
    rows = [
        {
            "source_type": "support_ticket",
            "support_ticket_cluster": "auto-alpha",
            "support_ticket_cluster_source": "token_set",
            "source_id": "auto-1",
            "source_title": "Provider metadata failed",
            "text": "Enterprise access fails after provider change",
        },
        {
            "source_type": "support_ticket",
            "support_ticket_cluster": "auto-beta",
            "support_ticket_cluster_source": "token_set",
            "source_id": "auto-2",
            "source_title": "Directory login failed",
            "text": "Company login error after directory update",
        },
    ]
    port = _PairEmbeddingPort()
    semantic_merges: list[dict[str, object]] = []

    result = build_ticket_faq_markdown(
        rows,
        max_items=0,
        embedding_port=port,
        embedding_merge_recorder=semantic_merges.append,
    )

    assert port.calls == [
        (
            "Enterprise access fails after provider change",
            "Company login error after directory update",
        )
    ]
    assert [
        (merge["left_source_id"], merge["right_source_id"])
        for merge in semantic_merges
    ] == [("auto-1", "auto-2")]
    assert len(result.items) == 1
    assert result.items[0]["ticket_count"] == 2
    assert result.items[0]["source_ids"] == ("auto-1", "auto-2")
    assert result.non_repeat_ticket_count == 0


def test_s5_cancel_subscription_and_order_intents_do_not_overmerge() -> None:
    package = build_support_ticket_input_package(
        [
            *_ticket_rows("sub", _CANCEL_SUBSCRIPTION_ROWS),
            *_ticket_rows("order", _CANCEL_ORDER_ROWS),
        ]
    )

    result = build_ticket_faq_markdown(package.inputs["source_material"], max_items=0)

    assert package.inputs["top_ticket_clusters"][0] == {"label": "order", "count": 17}
    assert len(result.items) == 3
    source_sets = [set(item["source_ids"]) for item in result.items]
    assert {"sub-1", "sub-6"} in source_sets
    assert {"order-1", "order-2", "order-6", "order-7"} in source_sets
    assert {"order-3", "order-9"} in source_sets
    assert all(
        not (
            any(source_id.startswith("sub-") for source_id in item["source_ids"])
            and any(source_id.startswith("order-") for source_id in item["source_ids"])
        )
        for item in result.items
    )
    assert result.non_repeat_ticket_count == 12


def test_s5_nested_advisory_cluster_source_is_inherited_before_topic_selection() -> None:
    result = build_ticket_faq_markdown(
        [
            {
                "source_type": "support_ticket",
                "support_ticket_cluster_source": "token_set",
                "evidence": [
                    {
                        "source_type": "support_ticket",
                        "support_ticket_cluster": "alpha",
                        "source_id": "alpha-1",
                        "text": "How do I reset password?",
                    },
                    {
                        "source_type": "support_ticket",
                        "support_ticket_cluster": "beta",
                        "source_id": "beta-1",
                        "text": "How do I reset password?",
                    },
                ],
            }
        ],
        max_items=0,
    )

    assert [(item["topic"], item["question"], item["source_ids"]) for item in result.items] == [
        ("login reset", "How do I reset password?", ("alpha-1", "beta-1"))
    ]


def test_s5_advisory_duplicate_can_join_matching_hard_topic() -> None:
    result = build_ticket_faq_markdown(
        [
            {
                "source_type": "support_ticket",
                "support_ticket_cluster": "invoice view",
                "support_ticket_cluster_source": "provided",
                "source_id": "provided-1",
                "text": "Where can I view my invoice?",
            },
            {
                "source_type": "support_ticket",
                "support_ticket_cluster": "invoice view",
                "support_ticket_cluster_source": "token_set",
                "source_id": "generated-1",
                "text": "Where can I view my invoice?",
            },
        ],
        max_items=0,
    )

    assert [(item["topic"], item["question"], item["source_ids"]) for item in result.items] == [
        ("invoice view", "Where can I view my invoice?", ("generated-1", "provided-1"))
    ]
    assert result.non_repeat_ticket_count == 0
    assert result.non_repeat_question_count == 0


def test_s5_output_order_and_source_id_order_are_input_order_stable() -> None:
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
        ("How do I get my money back?", ("refund-a1", "refund-a2")),
        ("Where can I request a refund credit?", ("refund-b1", "refund-b2")),
    ]


def test_s5_token_set_skip_leaves_large_preview_rows_uncategorized() -> None:
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
