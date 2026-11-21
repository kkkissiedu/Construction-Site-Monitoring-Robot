#!/bin/bash
# Gives the Jetson its own Wi-Fi network as a last-resort fallback, and sets the order it tries networks in.
# Run once on the Jetson (it asks for the sudo password):   bash setup_ap.sh
#   bash setup_ap.sh now      also switches to the new network straight away (drops the current Wi-Fi)
#   bash setup_ap.sh remove   deletes the network again
# Order of preference: home Wi-Fi, then any other saved network (for example a phone hotspot), then this one.
SSID="rover-jetson"
CON="rover-ap"
ADDRESS="10.42.0.1/24"       # the Jetson's fixed address on its own network
BAND="bg"                    # 2.4 GHz: the most compatible band
CHANNEL=6
PASS_FILE="$HOME/.rover_ap_password"

IFACE=$(nmcli -t -f DEVICE,TYPE dev | awk -F: '$2=="wifi"{print $1; exit}')
[ -n "$IFACE" ] || { echo "no Wi-Fi device found"; exit 1; }

sudo -v || exit 1

if [ "$1" = "remove" ]; then
    sudo nmcli con delete "$CON" && echo "removed $CON"
    exit 0
fi

# the password is made once and kept private; it never goes into the repo
if [ ! -s "$PASS_FILE" ]; then
    ( umask 077; tr -dc 'abcdefghjkmnpqrstuvwxyz23456789' < /dev/urandom | head -c 12 > "$PASS_FILE" )
fi
PASS=$(cat "$PASS_FILE")

command -v dnsmasq > /dev/null || echo "note: dnsmasq not found; NetworkManager may not hand out addresses (sudo apt install dnsmasq-base)"

sudo nmcli con delete "$CON" > /dev/null 2>&1
sudo nmcli con add type wifi ifname "$IFACE" con-name "$CON" ssid "$SSID" \
    connection.autoconnect yes connection.autoconnect-priority -10 \
    802-11-wireless.mode ap 802-11-wireless.band "$BAND" 802-11-wireless.channel "$CHANNEL" \
    802-11-wireless-security.key-mgmt wpa-psk 802-11-wireless-security.proto rsn \
    802-11-wireless-security.pairwise ccmp 802-11-wireless-security.group ccmp \
    802-11-wireless-security.psk "$PASS" \
    ipv4.method shared ipv4.addresses "$ADDRESS" ipv6.method ignore || exit 1

# the saved client networks: home Wi-Fi first, anything else after it, all above the Jetson's own network
nmcli -t -f NAME,TYPE con show | awk -F: '$2=="802-11-wireless"{print $1}' | while read -r name; do
    [ "$name" = "$CON" ] && continue
    case "$name" in
        HUAWEI*|Goliath*) prio=10 ;;
        *)                prio=5  ;;
    esac
    sudo nmcli con modify "$name" connection.autoconnect-priority "$prio" && echo "priority $prio: $name"
done

echo
echo "Network name : $SSID"
echo "Password     : $PASS   (also in $PASS_FILE)"
echo "Phone page   : http://${ADDRESS%/*}:8080/?key=$(cat "$HOME/.rover_remote_key" 2>/dev/null)"

if [ "$1" = "now" ]; then
    echo "switching to $SSID now..."
    sudo nmcli con up "$CON"
fi
