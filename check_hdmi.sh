#!/bin/bash
# Script to check and enable second HDMI port on RTX 4090

echo "=== Current Display Status ==="
DISPLAY=:0 xrandr --query | grep -E "^[A-Z].*:" | head -15

echo ""
echo "=== DRM Status ==="
for port in /sys/class/drm/card2-HDMI-A-*/status; do
    echo "$(basename $(dirname $port)): $(cat $port)"
done

echo ""
echo "=== Attempting to enable HDMI-1 ==="
# Try to enable the second HDMI port
DISPLAY=:0 xrandr --output HDMI-1 --auto 2>&1

echo ""
echo "=== Attempting to force 1080p on HDMI-1 ==="
# Try with a specific mode if auto doesn't work
DISPLAY=:0 xrandr --output HDMI-1 --mode 1920x1080 --rate 60 --right-of HDMI-0 2>&1

echo ""
echo "=== Final Status ==="
DISPLAY=:0 xrandr --listactivemonitors

echo ""
echo "=== Checking EDID ==="
for port in /sys/class/drm/card2-HDMI-A-*/edid; do
    size=$(stat -c%s "$port" 2>/dev/null || echo 0)
    echo "$(basename $(dirname $port)) EDID: $size bytes"
done

echo ""
echo "=== Troubleshooting Tips ==="
echo "1. Check that the HDMI cable is fully plugged in on both ends"
echo "2. Try unplugging and replugging the HDMI cable"
echo "3. Check if the monitor is set to the correct input source"
echo "4. Try a different HDMI cable if available"
echo "5. The monitor might need to send EDID data - try power cycling the monitor"
echo ""
echo "If EDID shows 0 bytes, the GPU cannot detect the monitor's capabilities."
echo "You may need to:"
echo "  - Replug the cable"
echo "  - Power cycle the monitor"
echo "  - Check cable quality (needs to support EDID/HDCP)"
echo "  - Try the other HDMI port to verify it's not a port issue"











