#!/bin/bash
# Diagnostic script for HDMI port issue on RTX 4090

echo "=========================================="
echo "HDMI Port Diagnostic for RTX 4090"
echo "=========================================="
echo ""

echo "=== Port Status Comparison ==="
echo "HDMI-A-2 (working port):"
echo "  Status: $(cat /sys/class/drm/card2-HDMI-A-2/status)"
echo "  Enabled: $(cat /sys/class/drm/card2-HDMI-A-2/enabled)"
echo "  DPMS: $(cat /sys/class/drm/card2-HDMI-A-2/dpms)"
echo "  Connector ID: $(cat /sys/class/drm/card2-HDMI-A-2/connector_id)"
echo ""
echo "HDMI-A-3 (non-working port):"
echo "  Status: $(cat /sys/class/drm/card2-HDMI-A-3/status)"
echo "  Enabled: $(cat /sys/class/drm/card2-HDMI-A-3/enabled)"
echo "  DPMS: $(cat /sys/class/drm/card2-HDMI-A-3/dpms)"
echo "  Connector ID: $(cat /sys/class/drm/card2-HDMI-A-3/connector_id)"
echo ""

echo "=== EDID Check ==="
size_a2=$(stat -c%s /sys/class/drm/card2-HDMI-A-2/edid 2>/dev/null || echo 0)
size_a3=$(stat -c%s /sys/class/drm/card2-HDMI-A-3/edid 2>/dev/null || echo 0)
echo "HDMI-A-2 EDID: $size_a2 bytes"
echo "HDMI-A-3 EDID: $size_a3 bytes"
echo ""

echo "=== Available Modes ==="
echo "HDMI-A-2 modes:"
cat /sys/class/drm/card2-HDMI-A-2/modes 2>/dev/null | head -5 || echo "  (none)"
echo ""
echo "HDMI-A-3 modes:"
cat /sys/class/drm/card2-HDMI-A-3/modes 2>/dev/null | head -5 || echo "  (none)"
echo ""

echo "=== xrandr Outputs ==="
DISPLAY=:0 xrandr --query | grep -E "^HDMI" | head -5
echo ""

echo "=== Diagnosis ==="
if [ "$(cat /sys/class/drm/card2-HDMI-A-3/status)" = "disconnected" ]; then
    echo "❌ HDMI-A-3 shows as 'disconnected'"
    echo ""
    echo "Since swapping monitors at the GPU end confirms:"
    echo "  ✓ Both monitors work (not a monitor issue)"
    echo "  ✓ Both cables work (not a cable issue)"
    echo "  ✗ One port doesn't detect connections (port hardware issue)"
    echo ""
    echo "This strongly suggests a HARDWARE PROBLEM with HDMI-A-3:"
    echo "  - Physical damage to the HDMI port"
    echo "  - Faulty port circuitry on the GPU"
    echo "  - Port not properly connected to the GPU board"
    echo ""
    echo "=== Possible Solutions ==="
    echo "1. Use DisplayPort instead of the second HDMI port"
    echo "2. Use an HDMI-to-DisplayPort adapter on HDMI-A-3"
    echo "3. Contact GPU manufacturer for RMA if under warranty"
    echo "4. Check if HDMI-A-3 works after:"
    echo "   - System reboot"
    echo "   - Reseating the GPU in the PCIe slot"
    echo "   - Checking GPU power connections"
    echo ""
fi











