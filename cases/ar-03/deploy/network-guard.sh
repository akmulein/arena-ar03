#!/bin/sh
set -eu
iptables -P OUTPUT DROP
# Docker rewrites its embedded DNS port before OUTPUT filtering.
iptables -A OUTPUT -d 127.0.0.11 -j REJECT
iptables -A OUTPUT -p udp --dport 53 -j REJECT
iptables -A OUTPUT -p tcp --dport 53 -j REJECT
iptables -A OUTPUT -o lo -j ACCEPT
iptables -A OUTPUT -m conntrack --ctstate ESTABLISHED,RELATED -j ACCEPT
iptables -A OUTPUT -d 172.29.83.20 -p tcp --dport 8080 -j ACCEPT
iptables -A OUTPUT -d 172.29.83.30 -p tcp --dport 3128 -j ACCEPT
ip6tables -P OUTPUT DROP
ip6tables -A OUTPUT -o lo -j ACCEPT
touch /tmp/network-ready
exec sleep infinity
