"""
Purchase confirmation email for Churn Signals vendor retention subscriptions.

Sent after a successful Stripe checkout. Table-based layout with inline CSS
for Outlook compatibility.
"""

from __future__ import annotations

from html import escape


_TIER_DETAILS = {
    "standard": {
        "name": "Standard",
        "price": "$499/mo",
        "features": [
            "Weekly churn intelligence reports",
            "Pain driver analysis",
            "Competitive displacement tracking",
            "Feature gap identification",
            "Anonymized customer signals",
        ],
    },
    "pro": {
        "name": "Pro",
        "price": "$1,499/mo",
        "features": [
            "Everything in Standard",
            "Named account-level signals",
            "Urgency scoring per account",
            "Real-time churn alerts",
            "Dedicated support",
        ],
    },
}


def render_checkout_confirmation_html(
    vendor_name: str,
    tier: str,
    customer_email: str,
) -> str:
    """Render the purchase confirmation email as HTML."""
    details = _TIER_DETAILS.get(tier, _TIER_DETAILS["standard"])
    vendor = escape(vendor_name)
    tier_name = details["name"]
    price = details["price"]

    feature_items = ""
    for feature in details["features"]:
        feature_items += (
            f'<tr><td style="padding:4px 0 4px 0;font-size:14px;color:#333;">'
            f'&#10003; {escape(feature)}</td></tr>'
        )

    return f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1.0"></head>
<body style="margin:0;padding:0;background:#f4f5f7;font-family:Arial,Helvetica,sans-serif;">
<table cellpadding="0" cellspacing="0" border="0" style="width:100%;background:#f4f5f7;">
<tr><td align="center" style="padding:24px 12px;">
<table cellpadding="0" cellspacing="0" border="0" style="width:100%;max-width:600px;background:#ffffff;border-radius:8px;overflow:hidden;box-shadow:0 2px 8px rgba(0,0,0,0.06);">

  <tr>
    <td style="background:#1a2332;padding:24px 24px 20px;text-align:center;">
      <div style="font-size:20px;font-weight:700;color:#ffffff;margin-bottom:4px;">Churn Signals</div>
      <div style="font-size:13px;color:#94a3b8;">Subscription Confirmed</div>
    </td>
  </tr>

  <tr>
    <td style="padding:28px 24px 20px;">
      <div style="font-size:22px;font-weight:700;color:#1a2332;margin-bottom:8px;">Thank you for subscribing</div>
      <p style="margin:0;font-size:15px;color:#444;line-height:1.6;">
        Your <strong>{vendor} {tier_name}</strong> subscription ({price}) is now active.
        Here is what to expect.
      </p>
    </td>
  </tr>

  <tr>
    <td style="padding:0 24px 20px;">
      <table cellpadding="0" cellspacing="0" border="0" style="width:100%;background:#f8fafc;border-radius:6px;border:1px solid #e2e8f0;">
        <tr>
          <td style="padding:16px 20px;">
            <div style="font-size:13px;font-weight:700;color:#2980b9;text-transform:uppercase;letter-spacing:0.05em;margin-bottom:12px;">What happens next</div>

            <table cellpadding="0" cellspacing="0" border="0" style="width:100%;">
              <tr>
                <td style="padding:0 0 10px;font-size:14px;color:#333;line-height:1.5;">
                  <strong style="color:#2980b9;">1.</strong> Your first full intelligence report will be delivered within <strong>24 hours</strong>.
                </td>
              </tr>
              <tr>
                <td style="padding:0 0 10px;font-size:14px;color:#333;line-height:1.5;">
                  <strong style="color:#2980b9;">2.</strong> After that, fresh reports arrive in your inbox <strong>every Monday</strong> morning.
                </td>
              </tr>
              <tr>
                <td style="padding:0;font-size:14px;color:#333;line-height:1.5;">
                  <strong style="color:#2980b9;">3.</strong> Each report covers pain drivers, competitive displacement, and actionable retention signals for {vendor}.
                </td>
              </tr>
            </table>
          </td>
        </tr>
      </table>
    </td>
  </tr>

  <tr>
    <td style="padding:0 24px 20px;">
      <table cellpadding="0" cellspacing="0" border="0" style="width:100%;background:#f8fafc;border-radius:6px;border:1px solid #e2e8f0;">
        <tr>
          <td style="padding:16px 20px;">
            <div style="font-size:13px;font-weight:700;color:#2980b9;text-transform:uppercase;letter-spacing:0.05em;margin-bottom:12px;">Your plan -- {tier_name} ({price})</div>
            <table cellpadding="0" cellspacing="0" border="0" style="width:100%;">
              {feature_items}
            </table>
          </td>
        </tr>
      </table>
    </td>
  </tr>

  <tr>
    <td style="padding:0 24px 20px;">
      <table cellpadding="0" cellspacing="0" border="0" style="width:100%;background:#f8fafc;border-radius:6px;border:1px solid #e2e8f0;">
        <tr>
          <td style="padding:16px 20px;">
            <div style="font-size:13px;font-weight:700;color:#2980b9;text-transform:uppercase;letter-spacing:0.05em;margin-bottom:12px;">Your account</div>
            <p style="margin:0;font-size:14px;color:#333;line-height:1.5;">
              You can sign in at
              <a href="https://churnsignals.co/login" style="color:#2980b9;text-decoration:none;font-weight:600;">churnsignals.co/login</a>
              to manage your subscription and access past reports.
            </p>
          </td>
        </tr>
      </table>
    </td>
  </tr>

  <tr>
    <td style="padding:0 24px 28px;">
      <p style="margin:0;font-size:14px;color:#555;line-height:1.6;">
        Have questions? Reply to this email or reach us at
        <a href="mailto:outreach@churnsignals.co" style="color:#2980b9;text-decoration:none;">outreach@churnsignals.co</a>.
      </p>
    </td>
  </tr>

  <tr>
    <td style="background:#f8fafc;padding:16px 24px;border-top:1px solid #e2e8f0;">
      <p style="margin:0;font-size:12px;color:#94a3b8;line-height:1.5;text-align:center;">
        Churn Signals by Atlas Business Intelligence<br>
        This is a transactional email confirming your purchase.
      </p>
    </td>
  </tr>

</table>
</td></tr>
</table>
</body>
</html>"""


def render_checkout_confirmation_text(
    vendor_name: str,
    tier: str,
) -> str:
    """Plain text fallback for the purchase confirmation email."""
    details = _TIER_DETAILS.get(tier, _TIER_DETAILS["standard"])
    features = "\n".join(f"  - {feature}" for feature in details["features"])
    return f"""Thank you for subscribing to Churn Signals

Your {vendor_name} {details['name']} subscription ({details['price']}) is now active.

WHAT HAPPENS NEXT
1. Your first full intelligence report will be delivered within 24 hours.
2. After that, fresh reports arrive in your inbox every Monday morning.
3. Each report covers pain drivers, competitive displacement, and actionable retention signals for {vendor_name}.

YOUR PLAN -- {details['name']} ({details['price']})
{features}

YOUR ACCOUNT
Sign in at https://churnsignals.co/login to manage your subscription and access past reports.

SUPPORT
Reply to this email or reach us at outreach@churnsignals.co.

--
Churn Signals by Atlas Business Intelligence
"""
