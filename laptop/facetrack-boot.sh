#!/bin/bash
# At login (autostart): the tracker window, then KWin places it bottom-left
# and keeps it above other windows (place.js).
c=$(dirname "$(readlink -f "$0")")
"$c/facetrack.sh" &
sleep 8
id=$(qdbus6 org.kde.KWin /Scripting org.kde.kwin.Scripting.loadScript "$c/place.js" ftplace 2>/dev/null)
qdbus6 org.kde.KWin "/Scripting/Script$id" org.kde.kwin.Script.run > /dev/null 2>&1
qdbus6 org.kde.KWin /Scripting org.kde.kwin.Scripting.unloadScript ftplace > /dev/null 2>&1
wait
