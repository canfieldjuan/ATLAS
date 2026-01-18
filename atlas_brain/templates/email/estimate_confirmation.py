"""
Email templates for Effingham Office Maids estimate confirmations.

Templates use Python string formatting with named placeholders.
"""

# Business contact info
BUSINESS_NAME = "Effingham Office Maids"
BUSINESS_ADDRESS = "503 S. 5th Street, Effingham IL, 62401"
BUSINESS_PHONE = "(217) 207-3097"
BUSINESS_EMAIL = "info@effinghamofficemaids.com"
BUSINESS_WEBSITE = "effinghamofficemaids.com"

# Key terms from proposal (to be prominently featured)
TERMS = (
    "Verbal authorization to perform the outlined work constitutes customer "
    "agreement to pay the quoted amount. Contractor not responsible for worn, "
    "faded, or damaged carpet or fabric."
)

# Business taglines
TAGLINES = [
    "We Clean Green Floor to Ceiling",
    "Affordable Prices",
    "Reliable Service",
    "Insured and Bonded",
    "Guaranteed Work",
]


# =============================================================================
# BUSINESS ESTIMATE CONFIRMATION TEMPLATE
# =============================================================================

BUSINESS_SUBJECT = "Cleaning Estimate Confirmation - {client_name}"

BUSINESS_TEMPLATE = """Dear {client_name},

Thank you for the opportunity to provide a cleaning estimate for your business. We appreciate your time and look forward to serving you.

ESTIMATE DETAILS
----------------
Service Location: {address}
Scheduled Date: {service_date}
Scheduled Time: {service_time}
Estimated Price: ${price}

ABOUT OUR SERVICE
-----------------
{BUSINESS_NAME} provides professional commercial cleaning services with a commitment to quality and reliability. We are fully insured and bonded for your peace of mind.

Our Promise:
- Affordable Prices
- Reliable Service
- Insured and Bonded
- Guaranteed Work
- We Clean Green Floor to Ceiling

NEXT STEPS
----------
To confirm this appointment, please reply to this email or contact us at {BUSINESS_PHONE}. Upon completion of services, an invoice will be provided with payment terms.

IMPORTANT TERMS
---------------
{TERMS}

If you have any questions or need to reschedule, please don't hesitate to reach out.

Best regards,

{BUSINESS_NAME}
{BUSINESS_ADDRESS}
Phone: {BUSINESS_PHONE}
Email: {BUSINESS_EMAIL}
Web: {BUSINESS_WEBSITE}
"""


# =============================================================================
# RESIDENTIAL ESTIMATE CONFIRMATION TEMPLATE
# =============================================================================

RESIDENTIAL_SUBJECT = "Your Cleaning Appointment Confirmation - {client_name}"

RESIDENTIAL_TEMPLATE = """Hi {client_name},

Thanks so much for choosing {BUSINESS_NAME}! We're excited to help make your home sparkle.

YOUR APPOINTMENT DETAILS
------------------------
Address: {address}
Date: {service_date}
Time: {service_time}
Estimated Cost: ${price}

WHAT TO EXPECT
--------------
Our friendly cleaning team will arrive at the scheduled time ready to work. We use eco-friendly products that are safe for your family and pets.

Need to reschedule? No problem! We understand life happens. Just give us a call at {BUSINESS_PHONE} or reply to this email, and we'll find a time that works better for you.

A FEW THINGS TO KNOW
--------------------
{TERMS}

WHY FAMILIES TRUST US
---------------------
- We Clean Green Floor to Ceiling
- Affordable Prices
- Reliable, Friendly Service
- Insured and Bonded
- 100% Satisfaction Guaranteed

Questions? We're here to help! Feel free to reach out anytime.

Looking forward to seeing you soon!

Warm regards,

The {BUSINESS_NAME} Team
{BUSINESS_PHONE}
{BUSINESS_EMAIL}
{BUSINESS_WEBSITE}
"""


def format_business_email(
    client_name: str,
    address: str,
    service_date: str,
    service_time: str,
    price: str,
) -> tuple[str, str]:
    """
    Format the business estimate confirmation email.

    Returns:
        Tuple of (subject, body)
    """
    subject = BUSINESS_SUBJECT.format(client_name=client_name)
    body = BUSINESS_TEMPLATE.format(
        client_name=client_name,
        address=address,
        service_date=service_date,
        service_time=service_time,
        price=price,
        BUSINESS_NAME=BUSINESS_NAME,
        BUSINESS_ADDRESS=BUSINESS_ADDRESS,
        BUSINESS_PHONE=BUSINESS_PHONE,
        BUSINESS_EMAIL=BUSINESS_EMAIL,
        BUSINESS_WEBSITE=BUSINESS_WEBSITE,
        TERMS=TERMS,
    )
    return subject, body


def format_residential_email(
    client_name: str,
    address: str,
    service_date: str,
    service_time: str,
    price: str,
) -> tuple[str, str]:
    """
    Format the residential estimate confirmation email.

    Returns:
        Tuple of (subject, body)
    """
    subject = RESIDENTIAL_SUBJECT.format(client_name=client_name)
    body = RESIDENTIAL_TEMPLATE.format(
        client_name=client_name,
        address=address,
        service_date=service_date,
        service_time=service_time,
        price=price,
        BUSINESS_NAME=BUSINESS_NAME,
        BUSINESS_PHONE=BUSINESS_PHONE,
        BUSINESS_EMAIL=BUSINESS_EMAIL,
        BUSINESS_WEBSITE=BUSINESS_WEBSITE,
        TERMS=TERMS,
    )
    return subject, body
