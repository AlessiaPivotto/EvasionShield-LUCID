base_dir="DATASETS/PCAP-CHUNKS/"

pcap_files=(
    "00-WebDDoS/webddos-chunk_chunk.pcap"
    "01-LDAP/ldap-chunk_chunk.pcap"
    "02-Portmap/portmap-chunk1_chunk.pcap"
    "03-DNS/dns-chunk2_chunk.pcap"
    "04-UDPLag/udplag-chunk_chunk.pcap"
    "05-NTP/ntp-chunk_chunk.pcap"
    "06-SNMP/snmp-chunk_chunk.pcap"
    "07-SSDP/ssdp-chunk_chunk.pcap"
    "08-Syn/syn-chunk6_chunk.pcap"
    "09-TFTP/tftp-chunk_chunk.pcap"
    "10-UDP/udp-chunk15_chunk.pcap"
    "11-NetBIOS/netbios-chunk_chunk.pcap"
    "12-MSSQL/mssql-chunk7_chunk.pcap"
)

# Manipulate all the pcaps in all the subfolders of base_dir
# pcap_files=($(find "$base_dir" -type f -name "*.pcap"))

for pcap_file in "${pcap_files[@]}"; do
    full_path="$base_dir/$pcap_file"
    # full_path="$pcap_file"
    if [[ -f "$full_path" ]]; then
        echo "Processing $full_path"
        python3 main.py -m "$full_path" -b example/mimic_set.npy -n example/normalizer.pkl -i example/init.pcap --fragment --fragment_prob 1.0 --min_fragment_size 64 --max_fragment_size 150
    else
        echo "File $full_path does not exist."
    fi
done

