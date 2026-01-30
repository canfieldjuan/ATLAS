"""
Response templates for edge devices.

Provides template-based response generation for common actions
and offline fallback messages.
"""

from typing import Optional

# Offline fallback message
OFFLINE_FALLBACK_MESSAGE = (
    "I can't have a full conversation right now - the brain is offline. "
    "I can still control your devices though."
)

# Brain unavailable message
BRAIN_UNAVAILABLE_MESSAGE = (
    "I'm having trouble reaching the brain server. "
    "Basic device commands still work."
)

# Device not found message template
DEVICE_NOT_FOUND_TEMPLATE = "I couldn't find a device called {name}."

# Action success templates
ACTION_SUCCESS_TEMPLATES = {
    "turn_on": [
        "Done.",
        "The {target} is now on.",
        "Turned on the {target}.",
    ],
    "turn_off": [
        "Done.",
        "The {target} is now off.",
        "Turned off the {target}.",
    ],
    "toggle": [
        "Done.",
        "Toggled the {target}.",
    ],
    "set_brightness": [
        "Done.",
        "Set the {target} to {brightness} percent.",
        "Brightness set to {brightness}%.",
    ],
    "brighten": [
        "Done.",
        "Brightened the {target}.",
    ],
    "dim": [
        "Done.",
        "Dimmed the {target}.",
    ],
    "play": [
        "Playing.",
        "Playing on {target}.",
    ],
    "pause": [
        "Paused.",
        "Paused the {target}.",
    ],
    "stop": [
        "Stopped.",
        "Stopped the {target}.",
    ],
    "mute": [
        "Muted.",
        "Muted the {target}.",
    ],
    "unmute": [
        "Unmuted.",
        "Unmuted the {target}.",
    ],
    "set_volume": [
        "Done.",
        "Volume set to {volume}%.",
    ],
    "volume_up": [
        "Volume up.",
    ],
    "volume_down": [
        "Volume down.",
    ],
}

# Action failure templates
ACTION_FAILURE_TEMPLATES = {
    "entity_not_found": "I couldn't find a device called {target}.",
    "unavailable": "The {target} doesn't seem to be available right now.",
    "timeout": "The {target} didn't respond in time.",
    "permission": "I don't have permission to control the {target}.",
    "default": "Sorry, I couldn't control the {target}.",
}

# Confirmation templates (for when user might want feedback)
CONFIRMATION_TEMPLATES = {
    "lights_on": "Lights are on.",
    "lights_off": "Lights are off.",
    "all_off": "Everything is off.",
}


class ResponseTemplates:
    """
    Template-based response generator for edge devices.

    Provides fast, consistent responses without LLM inference.
    """

    def __init__(self, use_short_responses: bool = True):
        """
        Initialize response templates.

        Args:
            use_short_responses: Use shorter "Done." style responses
        """
        self._use_short = use_short_responses

    def success_response(
        self,
        action: str,
        target: Optional[str] = None,
        **params,
    ) -> str:
        """
        Generate success response for an action.

        Args:
            action: Action performed
            target: Target device name
            **params: Additional parameters (brightness, volume, etc.)

        Returns:
            Response text
        """
        templates = ACTION_SUCCESS_TEMPLATES.get(action, ["Done."])

        # Use short or long response
        template = templates[0] if self._use_short else templates[-1]

        # Format with available values
        return template.format(
            target=target or "device",
            **params,
        )

    def failure_response(
        self,
        error_type: str,
        target: Optional[str] = None,
        error_message: Optional[str] = None,
    ) -> str:
        """
        Generate failure response.

        Args:
            error_type: Type of error
            target: Target device name
            error_message: Optional error message

        Returns:
            Response text
        """
        template = ACTION_FAILURE_TEMPLATES.get(
            error_type,
            ACTION_FAILURE_TEMPLATES["default"],
        )

        return template.format(target=target or "device")

    def offline_response(self, can_do_devices: bool = True) -> str:
        """
        Generate offline fallback response.

        Args:
            can_do_devices: Whether device commands still work

        Returns:
            Response text
        """
        if can_do_devices:
            return OFFLINE_FALLBACK_MESSAGE
        else:
            return "I'm sorry, I can't help right now. Please try again later."

    def device_not_found(self, name: str) -> str:
        """
        Generate device not found response.

        Args:
            name: Device name that wasn't found

        Returns:
            Response text
        """
        return DEVICE_NOT_FOUND_TEMPLATE.format(name=name)

    def confirmation(self, action_key: str) -> str:
        """
        Generate confirmation response.

        Args:
            action_key: Key for confirmation template

        Returns:
            Response text
        """
        return CONFIRMATION_TEMPLATES.get(action_key, "Done.")


# Singleton instance
_templates: Optional[ResponseTemplates] = None


def get_templates(use_short: bool = True) -> ResponseTemplates:
    """Get or create global templates."""
    global _templates
    if _templates is None:
        _templates = ResponseTemplates(use_short_responses=use_short)
    return _templates
