#!/bin/bash

base_dir="DATASETS/FLATTEN-PCAPS"
output_base_dir="PCAP-CHUNKS"

pcap_files=(
    "00-WebDDoS/webddos-chunk.pcap"
    "01-LDAP/ldap-chunk.pcap"
    "02-Portmap/portmap-chunk1.pcap"
    "03-DNS/dns-chunk2.pcap"
    "04-UDPLag/udplag-chunk.pcap"
    "05-NTP/ntp-chunk.pcap"
    "06-SNMP/snmp-chunk.pcap"
    "07-SSDP/ssdp-chunk.pcap"
    "08-Syn/syn-chunk6.pcap"
    "09-TFTP/tftp-chunk.pcap"
    "10-UDP/udp-chunk15.pcap"
    "11-NetBIOS/netbios-chunk.pcap"
    "12-MSSQL/mssql-chunk7.pcap"
)

for rel_path in "${pcap_files[@]}"; do
    input_path="${base_dir}/${rel_path}"

    # Extract filename and directory from relative path
    filename=$(basename "$rel_path" .pcap)
    subdir=$(dirname "$rel_path")

    # Set output directory structured by subdirectories
    output_dir="${output_base_dir}/${subdir}/${filename}"
    mkdir -p "$output_dir"

    echo "➤ Splitting $input_path into $output_dir"

    # Split file into 75MB chunks, outputs files like filename_chunk.pcap00, filename_chunk.pcap01, etc.
    tcpdump -r "$input_path" -C 75 -w "${output_dir}/${filename}_chunk.pcap"

    # Correctly rename files to place serial number before .pcap
    for f in "${output_dir}/${filename}_chunk.pcap"*; do
        [ -e "$f" ] || continue
        suffix="${f##*.pcap}" # get the numeric suffix
        mv "$f" "${output_dir}/${filename}_chunk${suffix}.pcap"
    done
done
