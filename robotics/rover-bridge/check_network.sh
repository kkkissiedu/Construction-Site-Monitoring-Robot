#!/bin/bash
# Shows how the Jetson is on the network and what to open on the phone. No sudo needed.
# Usage:   bash check_network.sh
IFACE=$(nmcli -t -f DEVICE,TYPE dev | awk -F: '$2=="wifi"{print $1; exit}')
ACTIVE=$(nmcli -t -f NAME,DEVICE con show --active | awk -F: -v d="$IFACE" '$2==d{print $1}')
MODE="client"
[ -n "$ACTIVE" ] && [ "$(nmcli -g 802-11-wireless.mode con show "$ACTIVE" 2>/dev/null)" = "ap" ] && MODE="access point"
IP=$(ip -4 -o addr show "$IFACE" 2>/dev/null | awk '{print $4}' | cut -d/ -f1 | head -1)
KEY=$(cat "$HOME/.rover_remote_key" 2>/dev/null)

echo "Wi-Fi device : $IFACE"
echo "Connection   : ${ACTIVE:-none} ($MODE)"
echo "Address      : ${IP:-none}"

if curl -s -m 3 "http://127.0.0.1:8080/ping?key=$KEY" | grep -q '"ok": true'; then
    echo "Rover remote : running on port 8080"
else
    echo "Rover remote : NOT answering on port 8080"
fi
[ -n "$IP" ] && echo "Phone page   : http://$IP:8080/?key=$KEY"

if [ "$MODE" = "access point" ]; then
    echo "Phones joined: $(iw dev "$IFACE" station dump 2>/dev/null | grep -c '^Station')"
    echo "Password     : in $HOME/.rover_ap_password"
fi

command -v dnsmasq > /dev/null && echo "dnsmasq      : found" || echo "dnsmasq      : not found (needed for the access point to hand out addresses)"
echo
AP_PRIO=$(nmcli -g connection.autoconnect-priority con show rover-ap 2>/dev/null)
if [ -z "$AP_PRIO" ]; then echo "Own network  : not set up (bash setup_ap.sh)"
elif [ "$AP_PRIO" -gt 10 ]; then echo "Own network  : on at every boot (priority $AP_PRIO)"
else echo "Own network  : only when no saved network is in range (priority $AP_PRIO)"; fi
echo
echo "Saved Wi-Fi networks (higher priority is tried first):"
nmcli -t -f NAME,TYPE,AUTOCONNECT-PRIORITY con show | awk -F: '$2=="802-11-wireless"{printf "  %-24s priority %s\n", $1, $3}'
