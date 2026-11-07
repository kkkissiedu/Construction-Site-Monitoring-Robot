#!/bin/bash
# One-time setup so the iPhone streams over USB: installs iproxy and the iPhone tools, then pairs the phone.
# Usage:  ssh -t orin 'bash ~/setup_usb_phone.sh'     (phone plugged into a USB-A port, unlocked)
set -e
sudo apt-get -o DPkg::Lock::Timeout=300 install -y libusbmuxd-tools libimobiledevice-utils
sudo systemctl restart usbmuxd
sleep 2
echo "Phone seen by usbmuxd:"
idevice_id -l || true
echo "Pairing: tap Trust on the phone, enter its passcode if asked."
idevicepair pair || true
idevicepair validate || true
