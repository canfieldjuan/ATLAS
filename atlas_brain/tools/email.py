"""
Email tool for sending emails via Resend API.

Allows Atlas to send emails on behalf of the user.
Ported from web-ui intelligent_email tool.
"""

import base64
import logging
from pathlib import Path
from typing import Any

import httpx

from ..config import settings
from ..templates.email import (
    format_business_email,
    format_residential_email,
    format_business_proposal,
    format_residential_proposal,
)
from .base import ToolParameter, ToolResult

logger = logging.getLogger("atlas.tools.email")


def is_path_in_whitelist(file_path: str) -> bool:
    """Check if file path is within whitelisted directories."""
    whitelist = settings.email.attachment_whitelist_dirs
    if not whitelist:
        return False

    resolved_path = Path(file_path).resolve()
    for allowed_dir in whitelist:
        allowed_path = Path(allowed_dir).resolve()
        try:
            resolved_path.relative_to(allowed_path)
            return True
        except ValueError:
            continue
    return False


def load_attachment(file_path: str) -> dict[str, str] | None:
    """
    Load and validate a file for email attachment.

    Returns dict with 'filename' and 'content' (base64) or None if invalid.
    """
    path = Path(file_path)

    # Check file exists
    if not path.exists():
        logger.warning("Attachment file not found: %s", file_path)
        return None

    # Check whitelist
    if not is_path_in_whitelist(file_path):
        logger.warning("Attachment path not in whitelist: %s", file_path)
        return None

    # Check file size
    max_bytes = settings.email.max_attachment_size_mb * 1024 * 1024
    file_size = path.stat().st_size
    if file_size > max_bytes:
        logger.warning(
            "Attachment too large: %s (%d bytes > %d max)",
            file_path, file_size, max_bytes
        )
        return None

    # Read and encode
    try:
        content = path.read_bytes()
        encoded = base64.b64encode(content).decode("utf-8")
        return {
            "filename": path.name,
            "content": encoded,
        }
    except Exception as e:
        logger.error("Failed to read attachment %s: %s", file_path, e)
        return None


def find_proposal_pdf(client_name: str) -> str | None:
    """
    Find a proposal PDF for the given client name.

    Searches all configured proposals_dirs for matching files.
    Returns file path if found, None otherwise.
    """
    proposals_dirs = settings.email.proposals_dirs
    if not proposals_dirs:
        return None

    # Search patterns (in order of preference)
    patterns = [
        f"{client_name} - Cleaning Proposal.pdf",
        f"{client_name} - Cleaning Proposal.PDF",
        f"{client_name}.pdf",
        f"{client_name}.PDF",
    ]

    for proposals_dir in proposals_dirs:
        base_path = Path(proposals_dir)
        if not base_path.exists():
            logger.debug("Proposals directory not found: %s", proposals_dir)
            continue

        # Try direct matches first
        for pattern in patterns:
            direct_path = base_path / pattern
            if direct_path.exists():
                logger.info("Found proposal PDF: %s", direct_path)
                return str(direct_path)

        # Try subdirectory match (client_name/*)
        client_dir = base_path / client_name
        if client_dir.exists() and client_dir.is_dir():
            for pdf_file in client_dir.glob("*.pdf"):
                if "proposal" in pdf_file.name.lower():
                    logger.info("Found proposal PDF in subdirectory: %s", pdf_file)
                    return str(pdf_file)
            # Return first PDF if no "proposal" in name
            pdf_files = list(client_dir.glob("*.pdf"))
            if pdf_files:
                logger.info("Found PDF in client directory: %s", pdf_files[0])
                return str(pdf_files[0])

    logger.debug("No proposal PDF found for client: %s", client_name)
    return None


class EmailTool:
    """Send emails via Resend API."""

    RESEND_API_URL = "https://api.resend.com/emails"

    def __init__(self) -> None:
        self._config = settings.email
        self._client: httpx.AsyncClient | None = None

    @property
    def name(self) -> str:
        return "send_email"

    @property
    def description(self) -> str:
        return (
            "Send an email on behalf of the user. "
            "Use this when the user asks you to email someone, "
            "send a message to an email address, or compose and send an email."
        )

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="to",
                param_type="string",
                description="Recipient email address (or comma-separated list)",
                required=True,
            ),
            ToolParameter(
                name="subject",
                param_type="string",
                description="Email subject line",
                required=True,
            ),
            ToolParameter(
                name="body",
                param_type="string",
                description="Email body content (plain text)",
                required=True,
            ),
            ToolParameter(
                name="from_email",
                param_type="string",
                description="Sender email address (uses default if not provided)",
                required=False,
            ),
            ToolParameter(
                name="cc",
                param_type="string",
                description="CC email addresses (comma-separated)",
                required=False,
            ),
            ToolParameter(
                name="bcc",
                param_type="string",
                description="BCC email addresses (comma-separated)",
                required=False,
            ),
            ToolParameter(
                name="reply_to",
                param_type="string",
                description="Reply-to email address",
                required=False,
            ),
        ]

    @property
    def aliases(self) -> list[str]:
        return ["email", "send email", "mail", "compose email"]

    @property
    def category(self) -> str:
        return "communication"

    async def _ensure_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=float(self._config.timeout))
        return self._client

    async def close(self) -> None:
        """Close HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def execute(self, params: dict[str, Any]) -> ToolResult:
        """Send email via Resend API."""
        # Check if enabled
        if not self._config.enabled:
            return ToolResult(
                success=False,
                error="TOOL_DISABLED",
                message="Email tool is disabled. Set ATLAS_EMAIL_ENABLED=true.",
            )

        # Check API key
        if not self._config.api_key:
            return ToolResult(
                success=False,
                error="NOT_CONFIGURED",
                message="Resend API key not configured. Set ATLAS_EMAIL_API_KEY.",
            )

        # Validate required parameters
        to = params.get("to")
        subject = params.get("subject")
        body = params.get("body")

        if not to:
            return ToolResult(
                success=False,
                error="MISSING_PARAMETER",
                message="Recipient email address (to) is required.",
            )

        if not subject:
            return ToolResult(
                success=False,
                error="MISSING_PARAMETER",
                message="Email subject is required.",
            )

        if not body:
            return ToolResult(
                success=False,
                error="MISSING_PARAMETER",
                message="Email body is required.",
            )

        # Get sender
        from_email = params.get("from_email") or self._config.default_from
        if not from_email:
            return ToolResult(
                success=False,
                error="NOT_CONFIGURED",
                message="No sender address. Set from_email or ATLAS_EMAIL_DEFAULT_FROM.",
            )

        # Parse recipients
        to_list = [email.strip() for email in to.split(",")]

        # Check recipient limit
        if len(to_list) > self._config.max_recipients:
            return ToolResult(
                success=False,
                error="LIMIT_EXCEEDED",
                message=f"Too many recipients. Maximum: {self._config.max_recipients}",
            )

        # Build payload
        payload: dict[str, Any] = {
            "from": from_email,
            "to": to_list,
            "subject": subject,
            "text": body,
        }

        # Optional fields
        if params.get("cc"):
            payload["cc"] = [email.strip() for email in params["cc"].split(",")]

        if params.get("bcc"):
            payload["bcc"] = [email.strip() for email in params["bcc"].split(",")]

        if params.get("reply_to"):
            payload["reply_to"] = params["reply_to"]

        # Handle attachments (list of file paths or pre-loaded attachment dicts)
        attachments = params.get("attachments", [])
        loaded_attachments = []
        for att in attachments:
            if isinstance(att, dict) and "filename" in att and "content" in att:
                # Already loaded attachment dict
                loaded_attachments.append(att)
            elif isinstance(att, str):
                # File path - load and validate
                loaded = load_attachment(att)
                if loaded:
                    loaded_attachments.append(loaded)
                else:
                    logger.warning("Skipping invalid attachment: %s", att)

        if loaded_attachments:
            payload["attachments"] = loaded_attachments
            logger.info("Adding %d attachment(s) to email", len(loaded_attachments))

        try:
            result = await self._send_email(payload)
            response_data = {
                "message_id": result.get("id"),
                "to": to_list,
                "subject": subject,
            }
            if loaded_attachments:
                response_data["attachments"] = [a["filename"] for a in loaded_attachments]

            return ToolResult(
                success=True,
                data=response_data,
                message=f"Email sent to {', '.join(to_list)}",
            )
        except httpx.HTTPStatusError as e:
            error_body = e.response.text
            logger.error("Resend API error: %s - %s", e.response.status_code, error_body)
            return ToolResult(
                success=False,
                error="API_ERROR",
                message=f"Failed to send email: {e.response.status_code}",
            )
        except Exception as e:
            logger.exception("Email tool error")
            return ToolResult(
                success=False,
                error="EXECUTION_ERROR",
                message=str(e),
            )

    async def _send_email(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Send email via Resend API."""
        client = await self._ensure_client()

        headers = {
            "Authorization": f"Bearer {self._config.api_key}",
            "Content-Type": "application/json",
        }

        logger.info(
            "Sending email to %s: %s",
            payload.get("to"),
            payload.get("subject", "")[:50],
        )

        response = await client.post(
            self.RESEND_API_URL,
            json=payload,
            headers=headers,
        )
        response.raise_for_status()

        result = response.json()
        logger.info("Email sent successfully: %s", result.get("id"))
        return result


# Module-level instance
email_tool = EmailTool()


class EstimateEmailTool:
    """Send cleaning estimate confirmation emails using templates."""

    def __init__(self) -> None:
        self._email_tool = email_tool

    @property
    def name(self) -> str:
        return "send_estimate_email"

    @property
    def description(self) -> str:
        return (
            "Send a cleaning estimate confirmation email to a client. "
            "Use this when confirming a cleaning appointment or sending an estimate. "
            "Automatically uses professional templates for business or residential clients."
        )

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="to",
                param_type="string",
                description="Client's email address",
                required=True,
            ),
            ToolParameter(
                name="client_name",
                param_type="string",
                description="Client's name (person or business name)",
                required=True,
            ),
            ToolParameter(
                name="address",
                param_type="string",
                description="Service address where cleaning will be performed",
                required=True,
            ),
            ToolParameter(
                name="service_date",
                param_type="string",
                description="Date of the cleaning service (e.g., 'January 20, 2026')",
                required=True,
            ),
            ToolParameter(
                name="service_time",
                param_type="string",
                description="Time of the cleaning service (e.g., '9:00 AM')",
                required=True,
            ),
            ToolParameter(
                name="price",
                param_type="string",
                description="Estimated price without dollar sign (e.g., '150.00')",
                required=True,
            ),
            ToolParameter(
                name="client_type",
                param_type="string",
                description="Type of client: 'business' for commercial or 'residential' for home",
                required=True,
            ),
        ]

    @property
    def aliases(self) -> list[str]:
        return ["estimate email", "send estimate", "cleaning estimate"]

    @property
    def category(self) -> str:
        return "communication"

    async def execute(self, params: dict[str, Any]) -> ToolResult:
        """Send estimate confirmation email using appropriate template."""
        # Validate required parameters
        required = ["to", "client_name", "address", "service_date", "service_time", "price", "client_type"]
        for param in required:
            if not params.get(param):
                return ToolResult(
                    success=False,
                    error="MISSING_PARAMETER",
                    message=f"Missing required parameter: {param}",
                )

        client_type = params["client_type"].lower().strip()
        if client_type not in ("business", "residential"):
            return ToolResult(
                success=False,
                error="INVALID_PARAMETER",
                message="client_type must be 'business' or 'residential'",
            )

        # Format email using appropriate template
        if client_type == "business":
            subject, body = format_business_email(
                client_name=params["client_name"],
                address=params["address"],
                service_date=params["service_date"],
                service_time=params["service_time"],
                price=params["price"],
            )
        else:
            subject, body = format_residential_email(
                client_name=params["client_name"],
                address=params["address"],
                service_date=params["service_date"],
                service_time=params["service_time"],
                price=params["price"],
            )

        # Send via the base email tool
        email_params = {
            "to": params["to"],
            "subject": subject,
            "body": body,
            "reply_to": "info@effinghamofficemaids.com",
        }

        result = await self._email_tool.execute(email_params)

        if result.success:
            result.message = (
                f"Estimate confirmation sent to {params['client_name']} "
                f"({params['to']}) for {params['service_date']}"
            )
            result.data["template"] = client_type
            result.data["client_name"] = params["client_name"]

        return result


# Module-level instance
estimate_email_tool = EstimateEmailTool()


class ProposalEmailTool:
    """Send cleaning proposal emails using templates."""

    def __init__(self) -> None:
        self._email_tool = email_tool

    @property
    def name(self) -> str:
        return "send_proposal_email"

    @property
    def description(self) -> str:
        return (
            "Send a cleaning proposal email to a potential client after an estimate visit. "
            "Use this when sending a detailed proposal with areas to clean, services offered, "
            "and pricing. Works for both business and residential clients."
        )

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="to",
                param_type="string",
                description="Client's email address",
                required=True,
            ),
            ToolParameter(
                name="client_name",
                param_type="string",
                description="Business name or family name",
                required=True,
            ),
            ToolParameter(
                name="contact_name",
                param_type="string",
                description="Contact person's name",
                required=True,
            ),
            ToolParameter(
                name="address",
                param_type="string",
                description="Service address",
                required=True,
            ),
            ToolParameter(
                name="areas_to_clean",
                param_type="string",
                description="Areas to be cleaned (e.g., 'Offices, Bathrooms, Break Room' or '3 Bedrooms, 2 Bathrooms')",
                required=True,
            ),
            ToolParameter(
                name="cleaning_description",
                param_type="string",
                description="Description of cleaning tasks (e.g., 'Dust and disinfect surfaces, vacuum floors, empty trash')",
                required=True,
            ),
            ToolParameter(
                name="price",
                param_type="string",
                description="Price per cleaning without dollar sign (e.g., '150.00')",
                required=True,
            ),
            ToolParameter(
                name="frequency",
                param_type="string",
                description="Cleaning frequency (e.g., 'Weekly', 'Bi-weekly', 'Monthly', 'As needed')",
                required=False,
            ),
            ToolParameter(
                name="contact_phone",
                param_type="string",
                description="Contact phone number (required for business proposals)",
                required=False,
            ),
            ToolParameter(
                name="client_type",
                param_type="string",
                description="Type of client: 'business' for commercial or 'residential' for home",
                required=True,
            ),
        ]

    @property
    def aliases(self) -> list[str]:
        return ["proposal email", "send proposal", "cleaning proposal"]

    @property
    def category(self) -> str:
        return "communication"

    async def execute(self, params: dict[str, Any]) -> ToolResult:
        """Send proposal email using appropriate template."""
        # Validate required parameters
        required = ["to", "client_name", "contact_name", "address", "areas_to_clean", "cleaning_description", "price", "client_type"]
        for param in required:
            if not params.get(param):
                return ToolResult(
                    success=False,
                    error="MISSING_PARAMETER",
                    message=f"Missing required parameter: {param}",
                )

        client_type = params["client_type"].lower().strip()
        if client_type not in ("business", "residential"):
            return ToolResult(
                success=False,
                error="INVALID_PARAMETER",
                message="client_type must be 'business' or 'residential'",
            )

        # Business proposals require contact_phone
        if client_type == "business" and not params.get("contact_phone"):
            return ToolResult(
                success=False,
                error="MISSING_PARAMETER",
                message="contact_phone is required for business proposals",
            )

        frequency = params.get("frequency", "As needed")

        # Format email using appropriate template
        if client_type == "business":
            subject, body = format_business_proposal(
                client_name=params["client_name"],
                contact_name=params["contact_name"],
                contact_phone=params["contact_phone"],
                address=params["address"],
                areas_to_clean=params["areas_to_clean"],
                cleaning_description=params["cleaning_description"],
                price=params["price"],
                frequency=frequency,
            )
        else:
            subject, body = format_residential_proposal(
                client_name=params["client_name"],
                contact_name=params["contact_name"],
                address=params["address"],
                areas_to_clean=params["areas_to_clean"],
                cleaning_description=params["cleaning_description"],
                price=params["price"],
                frequency=frequency,
            )

        # Auto-find proposal PDF attachment
        attachments = []
        pdf_path = find_proposal_pdf(params["client_name"])
        if pdf_path:
            attachments.append(pdf_path)
            logger.info("Auto-attaching proposal PDF: %s", pdf_path)

        # Send via the base email tool
        email_params = {
            "to": params["to"],
            "subject": subject,
            "body": body,
            "reply_to": "info@effinghamofficemaids.com",
        }
        if attachments:
            email_params["attachments"] = attachments

        result = await self._email_tool.execute(email_params)

        if result.success:
            msg = f"Proposal sent to {params['client_name']} ({params['to']}) - ${params['price']} {frequency}"
            if pdf_path:
                msg += " [PDF attached]"
            result.message = msg
            result.data["template"] = client_type
            result.data["client_name"] = params["client_name"]
            if pdf_path:
                result.data["pdf_attached"] = True

        return result


# Module-level instance
proposal_email_tool = ProposalEmailTool()
