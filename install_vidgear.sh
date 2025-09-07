#!/bin/bash

echo "🚀 Installing VidGear for high-performance RTSP capture..."
pip install vidgear[core]

echo ""
echo "✅ VidGear installation complete!"
echo ""
echo "📝 VidGear provides:"
echo "   - Multi-threaded RTSP capture"  
echo "   - 10+ FPS performance"
echo "   - Built-in error handling"
echo "   - Superior to FFmpeg subprocess"
echo "   - Solves frame-120 freeze issue"
echo ""
echo "🔄 Restart your application to use VidGear!"
echo ""
echo "Alternative manual installation:"
echo "   pip install vidgear"
echo "   # or with extras:"
echo "   pip install vidgear[asyncio,core]"

