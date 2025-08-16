base_dir="PCAP-CHUNKS"

pcap_files=(

    "09-TFTP/tftp-chunk9/tftp-chunk9_chunk.pcap"
    "09-TFTP/tftp-chunk9/tftp-chunk9_chunk1.pcap"
    "09-TFTP/tftp-chunk9/tftp-chunk9_chunk2.pcap"
    "09-TFTP/tftp-chunk9/tftp-chunk9_chunk3.pcap"
    "09-TFTP/tftp-chunk9/tftp-chunk9_chunk4.pcap"
    "09-TFTP/tftp-chunk9/tftp-chunk9_chunk5.pcap"
    "09-TFTP/tftp-chunk9/tftp-chunk9_chunk6.pcap"
    "09-TFTP/tftp-chunk9/tftp-chunk9_chunk7.pcap"
    "09-TFTP/tftp-chunk9/tftp-chunk9_chunk8.pcap"
    "09-TFTP/tftp-chunk9/tftp-chunk9_chunk9.pcap"
    "09-TFTP/tftp-chunk9/tftp-chunk9_chunk10.pcap"
    "09-TFTP/tftp-chunk9/tftp-chunk9_chunk11.pcap"
    "09-TFTP/tftp-chunk9/tftp-chunk9_chunk12.pcap"
    "09-TFTP/tftp-chunk9/tftp-chunk9_chunk13.pcap"
    "09-TFTP/tftp-chunk9/tftp-chunk9_chunk14.pcap"
    "09-TFTP/tftp-chunk9/tftp-chunk9_chunk15.pcap"
    "09-TFTP/tftp-chunk9/tftp-chunk9_chunk16.pcap"
    "09-TFTP/tftp-chunk9/tftp-chunk9_chunk17.pcap"
    "09-TFTP/tftp-chunk9/tftp-chunk9_chunk18.pcap"
    "09-TFTP/tftp-chunk9/tftp-chunk9_chunk19.pcap"
    "09-TFTP/tftp-chunk9/tftp-chunk9_chunk20.pcap"
    "09-TFTP/tftp-chunk9/tftp-chunk9_chunk21.pcap"
    "09-TFTP/tftp-chunk9/tftp-chunk9_chunk22.pcap"
    "09-TFTP/tftp-chunk9/tftp-chunk9_chunk23.pcap"
    "09-TFTP/tftp-chunk9/tftp-chunk9_chunk24.pcap"
    "09-TFTP/tftp-chunk9/tftp-chunk9_chunk25.pcap"
    "09-TFTP/tftp-chunk9/tftp-chunk9_chunk26.pcap"

)

# Manipulate all the pcaps in all the subfolders of base_dir
# pcap_files=($(find "$base_dir" -type f -name "*.pcap"))

for pcap_file in "${pcap_files[@]}"; do
    full_path="$base_dir/$pcap_file"
    # full_path="$pcap_file"
    if [[ -f "$full_path" ]]; then
        echo "Processing $full_path"
        python3 main.py -m "$full_path" -b example/mimic_set.npy -n example/normalizer.pkl -i example/init.pcap
    else
        echo "File $full_path does not exist."
    fi
done

