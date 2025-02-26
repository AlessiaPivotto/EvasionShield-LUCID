import os
from scapy.all import rdpcap, wrpcap
from tqdm import tqdm

def merge_pcap_files(input_folder):
    packets = []
    tqdm.write(f"Merging pcap files in {input_folder}")
    
    pcap_files = [f for f in os.listdir(input_folder) if f.endswith(".pcap")]

    for filename in tqdm(pcap_files, desc="Processing pcap files"):
        file_path = os.path.join(input_folder, filename)
        packets.extend(rdpcap(file_path))
    
    output_file = os.path.join("dataset/", os.path.basename(input_folder) + ".pcap")
    wrpcap(output_file, packets)
    print(f"Saved {len(packets)} packets to {output_file}")

if __name__ == "__main__":

    input_folder = "dataset/UDPLag"
    merge_pcap_files(input_folder)