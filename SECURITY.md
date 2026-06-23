# Security Policy

## Reporting Vulnerabilities

Report suspected vulnerabilities through GitHub private vulnerability reporting:

https://github.com/canfieldjuan/ATLAS/security/advisories/new

Do not open a public GitHub issue for exploitable vulnerabilities, exposed
credentials, authentication bypasses, payment or billing issues, data deletion
gaps, or report-access bugs.

Include as much of the following as you can:

- Affected component, route, workflow, or repository.
- Steps to reproduce the issue.
- Security impact and any data class involved.
- Sanitized logs, screenshots, proof-of-concept requests, or affected IDs.
- Whether the issue crosses ATLAS and atlas-portfolio surfaces.

## Scope

This policy covers the ATLAS repository, including API services, workflow
automation, authentication and authorization controls, payment-adjacent flows,
data deletion and retention behavior, security scanning, and generated customer
artifacts.

For cross-repository deflection funnel findings, use the same reporting path and
name both ATLAS and atlas-portfolio in the report.

## Researcher Guidelines

Good-faith testing is welcome when it avoids harm:

- Do not destroy, alter, or exfiltrate data beyond the minimum proof needed.
- Do not persist access after demonstrating impact.
- Do not test denial-of-service, spam, social engineering, or physical attacks.
- Do not access data belonging to other users except when strictly necessary to
  prove an authorization boundary issue, and stop immediately after proof.
- Keep vulnerability details private until the issue is resolved or disclosure is
  coordinated.

Reports that follow these guidelines are treated as authorized security research.

## Incident Response

Operational triage and response steps live in
[docs/INCIDENT_RESPONSE.md](docs/INCIDENT_RESPONSE.md).

## Response Targets

Atlas aims to acknowledge new vulnerability reports within five business days,
confirm scope and impact after triage, and provide status updates for accepted
reports until remediation or documented acceptance.

Specific remediation deadlines and CVE handling are tracked separately from this
initial disclosure policy.
