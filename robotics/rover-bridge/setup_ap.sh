#!/bin/bash
# The Jetson's own Wi-Fi network ("rover-jetson") and the order the Jetson tries networks in. Run on the Jetson;
# it asks for the sudo password. Re-running is safe and does not drop a connected phone.
#   bash setup_ap.sh              own network is on at every boot (the default, "rover" mode)
#   bash setup_ap.sh fallback     own network only when no saved network is in range
#   bash setup_ap.sh online       own network off; join the best saved network now (for apt, pip, updates)
#   bash setup_ap.sh hotspot      own network off; join the saved iPhone hotspot now and prefer it over home Wi-Fi
#                                 (the sensor phone then sends over its own fast 5 GHz hotspot; "rover" mode undoes it)
#   bash setup_ap.sh remove       delete the own network
# Add "now" to switch immediately:  bash setup_ap.sh rover now   (the Jetson leaves its current Wi-Fi)
SSID="rover-jetson"
CON="rover-ap"
ADDRESS="10.42.0.1/24"       # the Jetson's fixed address on its own network
BAND="bg"                    # 2.4 GHz: the most compatible band
CHANNEL=6
PASS_FILE="$HOME/.rover_ap_password"

MODE="${1:-rover}"
NOW="$2"
[ "$MODE" = "now" ] && { MODE="rover"; NOW="now"; }

IFACE=$(nmcli -t -f DEVICE,TYPE dev | awk -F: '$2=="wifi"{print $1; exit}')
[ -n "$IFACE" ] || { echo "no Wi-Fi device found"; exit 1; }

sudo -v || exit 1

exists() { nmcli -t -f NAME con show | grep -qx "$CON"; }

# saved client networks, best first
set_client_priorities() {
    nmcli -t -f NAME,TYPE con show | awk -F: '$2=="802-11-wireless"{print $1}' | while read -r name; do
        [ "$name" = "$CON" ] && continue
        case "$name" in
            HUAWEI*|Goliath*) prio=10 ;;
            *)                prio=5  ;;
        esac
        sudo nmcli con modify "$name" connection.autoconnect-priority "$prio"
    done
}

case "$MODE" in
    remove)
        sudo nmcli con delete "$CON" && echo "removed $CON"
        exit 0
        ;;
    online)
        exists && sudo nmcli con modify "$CON" connection.autoconnect-priority -10
        set_client_priorities
        echo "own network is off from now on; trying the saved networks, best first"
        nmcli -t -f NAME,TYPE,AUTOCONNECT-PRIORITY con show | awk -F: '$2=="802-11-wireless" && $1!="'"$CON"'"{print $3":"$1}' \
            | sort -t: -k1 -nr | cut -d: -f2- | while read -r name; do
            sudo nmcli con up "$name" > /dev/null 2>&1 && { echo "connected to $name"; exit 0; }
        done
        exit 0
        ;;
    hotspot)
        exists && sudo nmcli con modify "$CON" connection.autoconnect-priority -10
        set_client_priorities
        phone=$(nmcli -t -f NAME,TYPE con show | awk -F: '$2=="802-11-wireless" && /iPhone/{print $1; exit}')
        [ -n "$phone" ] || { echo "no saved iPhone hotspot (join it once from the Jetson first)"; exit 1; }
        sudo nmcli con modify "$phone" connection.autoconnect-priority 20
        echo "own network is off; joining $phone (turn its Personal Hotspot on first)"
        sudo nmcli con up "$phone" || exit 1
        echo "Jetson address: $(ip -4 -o addr show "$IFACE" | awk '{print $4}' | cut -d/ -f1)"
        exit 0
        ;;
    rover)    AP_PRIO=100 ;;
    fallback) AP_PRIO=-10 ;;
    *) echo "unknown mode: $MODE"; exit 1 ;;
esac

# the password is made once and kept private; it never goes into the repo
if [ ! -s "$PASS_FILE" ]; then
    ( umask 077; tr -dc 'abcdefghjkmnpqrstuvwxyz23456789' < /dev/urandom | head -c 12 > "$PASS_FILE" )
fi
PASS=$(cat "$PASS_FILE")

command -v dnsmasq > /dev/null || echo "note: dnsmasq not found; NetworkManager may not hand out addresses (sudo apt install dnsmasq-base)"

SETTINGS=(
    connection.autoconnect yes connection.autoconnect-priority "$AP_PRIO"
    802-11-wireless.ssid "$SSID" 802-11-wireless.mode ap 802-11-wireless.band "$BAND" 802-11-wireless.channel "$CHANNEL"
    802-11-wireless-security.key-mgmt wpa-psk 802-11-wireless-security.proto rsn
    802-11-wireless-security.pairwise ccmp 802-11-wireless-security.group ccmp
    802-11-wireless-security.psk "$PASS"
    ipv4.method shared ipv4.addresses "$ADDRESS" ipv6.method ignore
)
if exists; then
    sudo nmcli con modify "$CON" "${SETTINGS[@]}" || exit 1       # changes apply the next time it starts
else
    sudo nmcli con add type wifi ifname "$IFACE" con-name "$CON" "${SETTINGS[@]}" || exit 1
fi
set_client_priorities

echo
echo "Mode         : $MODE (own network priority $AP_PRIO)"
echo "Network name : $SSID"
echo "Password     : $PASS   (also in $PASS_FILE)"
echo "Phone page   : http://${ADDRESS%/*}:8080/?key=$(cat "$HOME/.rover_remote_key" 2>/dev/null)"

if [ "$NOW" = "now" ]; then
    echo "switching to $SSID now..."
    sudo nmcli con up "$CON"
fi
