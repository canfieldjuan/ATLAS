# Node SSH Access (Tailscale)

Last updated: 2026-02-18

## Change Log

- 2026-02-18: No change to SSH transport, hostnames, user, or command flow.
- 2026-02-18: Security changes were applied to dashboard/API/WebRTC auth only.
- 2026-02-18: Continue using Tailscale SSH exactly as before.

## Requirements

1. Tailscale is installed and logged in on the machine you are connecting from.
2. The node is online in the same Tailscale tailnet.
3. You have SSH access for `canfieldjuan` on the node.

## Connect Commands

```bash
# Optional: verify node presence first
tailscale status | grep -E 'orangepi-atlas|100.95.224.113'
tailscale ping orangepi-atlas

# Preferred: Tailscale hostname
ssh canfieldjuan@orangepi-atlas

# Fallback: Tailscale IP
ssh canfieldjuan@100.95.224.113
```

## Post-Login Sanity Checks

```bash
hostname
cd /opt/atlas-node
git status --short --branch
systemctl is-active atlas-node.service
```

## Troubleshooting

```bash
# Resolve current Tailscale IP for hostname
tailscale ip -4 orangepi-atlas

# If host key changed:
ssh-keygen -R orangepi-atlas
ssh-keygen -R 100.95.224.113
```

## Important Scope Note

- SSH access to the node shell is unchanged.
- Dashboard/API endpoints require token auth.
- MediaMTX stream reads (WebRTC/HLS) now require credentials.
