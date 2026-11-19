#!/bin/bash
# Makes the rover software start by itself at boot (user cron, no sudo). Run once on the Jetson:  bash install_autostart.sh
# Undo with:  bash install_autostart.sh remove
DIR="$(cd "$(dirname "$0")" && pwd)"
TAG="# rover-autostart"
LINE="@reboot sleep 15 && /bin/bash $DIR/rover_autostart.sh $TAG"

OLD="$(crontab -l 2>/dev/null | grep -v "$TAG")"
if [ "$1" = "remove" ]; then
    echo "$OLD" | crontab -
    echo "autostart removed"
else
    printf '%s\n%s\n' "$OLD" "$LINE" | sed '/^$/d' | crontab -
    echo "autostart installed:"; crontab -l | grep "$TAG"
fi
