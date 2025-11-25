from scapy.all import *
from scapy.layers.inet import IP, TCP, UDP, ICMP
from scapy.layers.l2 import Ether, ARP
from scapy.layers.inet6 import IPv6
import random
import string
import numpy as np


def random_bytes(length):
    tmp_str = ''.join(random.choice(string.printable) for _ in range(length))
    return bytes(tmp_str, encoding='utf-8')


def fragment_packet(pkt, min_fragment_size=64, max_fragment_size=1200):
    """
    Fragment an IP packet into random-sized fragments with randomized fragment sizes.
    
    This function implements intelligent fragmentation that:
    - Uses random fragment sizes between min and max bounds
    - Ensures fragments are at least min_fragment_size (default 64 bytes)
    - Maintains proper IP fragmentation headers (MF flag, fragment offset)
    - Preserves packet timing and other metadata
    - Uses proper 8-byte alignment for fragment offsets (except last fragment)
    
    Args:
        pkt: The scapy packet to fragment
        min_fragment_size: Minimum fragment size in bytes (default: 64)
        max_fragment_size: Maximum fragment size in bytes (default: 1200)
        
    Returns:
        List of fragmented packets or [original_packet] if no fragmentation needed
    """
    if not pkt.haslayer(IP):
        return [pkt]
    
    # Get the IP layer and its payload
    ip_layer = pkt[IP]
    
    # Calculate the actual payload (everything after IP header)
    # Handle case where ihl might be None (auto-calculated by scapy)
    if ip_layer.ihl is None:
        ip_header_len = 20  # Standard IP header length
    else:
        ip_header_len = ip_layer.ihl * 4  # IP header length in bytes
    
    total_packet_size = len(pkt)
    payload_size = total_packet_size - ip_header_len
    
    # If packet is too small to fragment meaningfully, return as-is
    if payload_size <= min_fragment_size:
        return [pkt]
    
    # Extract payload bytes (everything after IP header)
    payload = bytes(pkt)[ip_header_len:]
    
    fragments = []
    payload_offset = 0
    fragment_id = random.randint(1, 65535)  # Random fragment ID
    original_time = pkt.time
    
    print(f"    Fragmenting packet of {payload_size} bytes into random-sized fragments")
    
    while payload_offset < len(payload):
        # Calculate remaining payload
        remaining_payload = len(payload) - payload_offset
        
        # Determine if we can create at least one more fragment after this one
        can_create_another = remaining_payload > (2 * min_fragment_size)
        
        # Determine fragment size with randomization
        if not can_create_another:
            # This will be the last fragment - use all remaining data
            fragment_payload_size = remaining_payload
            more_fragments = False
        else:
            # We can create another fragment, so randomize this fragment's size
            # Ensure we don't create a final fragment smaller than min_fragment_size
            max_allowed_size = min(max_fragment_size, 
                                 remaining_payload - min_fragment_size)
            
            if max_allowed_size < min_fragment_size:
                # Edge case: just use remaining payload
                fragment_payload_size = remaining_payload
                more_fragments = False
            else:
                # Random size within bounds
                fragment_payload_size = random.randint(min_fragment_size, max_allowed_size)
                more_fragments = True
        
        # Ensure fragment payload size is multiple of 8 bytes (except for last fragment)
        # This is required by IP fragmentation specification
        if more_fragments and fragment_payload_size % 8 != 0:
            fragment_payload_size = (fragment_payload_size // 8) * 8
            # Ensure we didn't go below minimum
            if fragment_payload_size < min_fragment_size:
                fragment_payload_size = ((min_fragment_size + 7) // 8) * 8
        
        # Create the fragment packet
        fragment_pkt = IP()
        
        # Copy all IP header fields from original
        fragment_pkt.version = ip_layer.version
        fragment_pkt.ihl = ip_layer.ihl if ip_layer.ihl is not None else 5  # Default to 5 (20 bytes)
        fragment_pkt.tos = ip_layer.tos
        fragment_pkt.ttl = ip_layer.ttl
        fragment_pkt.proto = ip_layer.proto
        fragment_pkt.src = ip_layer.src
        fragment_pkt.dst = ip_layer.dst
        if hasattr(ip_layer, 'options'):
            fragment_pkt.options = ip_layer.options
        
        # Set fragment-specific fields
        fragment_pkt.id = fragment_id
        fragment_pkt.flags = 'MF' if more_fragments else 0  # More Fragments flag
        fragment_pkt.frag = payload_offset // 8  # Fragment offset in 8-byte units
        
        # Add the fragment payload
        fragment_payload = payload[payload_offset:payload_offset + fragment_payload_size]
        fragment_pkt = fragment_pkt / Raw(load=fragment_payload)
        
        # Calculate and set correct length
        fragment_pkt.len = len(fragment_pkt)
        
        # Preserve original packet metadata
        fragment_pkt.time = original_time
        
        # Copy any lower layer headers (Ethernet, etc.)
        if pkt.haslayer(Ether):
            eth_header = pkt[Ether].copy()
            eth_header.remove_payload()
            fragment_pkt = eth_header / fragment_pkt
        
        fragments.append(fragment_pkt)
        
        print(f"      Fragment {len(fragments)}: offset={payload_offset}, size={fragment_payload_size}, "
              f"MF={'Yes' if more_fragments else 'No'}")
        
        payload_offset += fragment_payload_size
    
    print(f"    Created {len(fragments)} fragments from original packet")
    return fragments
def rebuild(
    grp_size,
    X,
    groupList,
    # tmp_pcap_file
):

    newList = []

    for i in range(grp_size):

        for j in range(int(round(X.mal[i][1]))):
            pkt = copy.deepcopy(groupList[i])
            if round(X.craft[i][j][1]) == 1:
                if groupList[i].haslayer(Ether):
                    pkt[Ether].remove_payload()
                else:
                    raise RuntimeError("Error in rebuilder!")

            elif round(X.craft[i][j][1]) == 2:
                if groupList[i].haslayer(IP):
                    pkt[IP].remove_payload()
                elif groupList[i].haslayer(IPv6):
                    pkt[IPv6].remove_payload()
                elif groupList[i].haslayer(ARP):
                    pkt[ARP].remove_payload()
                else:
                    raise RuntimeError("Error in rebuilder!")
            elif round(X.craft[i][j][1]) == 3:
                if groupList[i].haslayer(ICMP):
                    pkt[ICMP].remove_payload()
                elif groupList[i].haslayer(TCP):
                    pkt[TCP].remove_payload()
                elif groupList[i].haslayer(UDP):
                    pkt[UDP].remove_payload()
                else:
                    raise RuntimeError("Error in rebuilder!")
            else:
                raise RuntimeError("Error in rebuilder!")
            pkt.add_payload(random_bytes(int(round(X.craft[i][j][2]))))
            pkt.time = X.mal[i][0] - X.craft[i][j][0]
            newList.append(pkt)

        mal_pkt = copy.deepcopy(groupList[i])
        mal_pkt.time = X.mal[i][0]
        newList.append(mal_pkt)

    # wrpcap(tmp_pcap_file, newList)
    return newList
