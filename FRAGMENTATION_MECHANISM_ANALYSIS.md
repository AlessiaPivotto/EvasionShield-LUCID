# Stochastic IP Fragmentation as an Adversarial Evasion Mechanism: Implementation Analysis

## Abstract

This document presents a comprehensive analysis of the packet fragmentation mechanism implemented within the `TrafficManipulator` framework. It details the algorithmic approach designed to obfuscate traffic patterns through stochastic decomposition of IP payloads. By leveraging compliant yet randomized fragmentation, the system generates valid network traffic that diverges significantly from the statistical profile of the original flow, thereby challenging flow-based Intrusion Detection Systems (IDS) and Machine Learning classifiers while maintaining protocol validity.

## 1. Introduction

Packet fragmentation is a fundamental feature of the Internet Protocol (IP) designed to allow the transmission of datagrams across networks with varying Maximum Transmission Units (MTU). However, in the context of adversarial machine learning and network evasion, fragmentation serves as a perturbation technique. By splitting a single logical packet into multiple smaller physical frames, an attacker can alter features such as packet size distribution, inter-arrival times, and payload signatures without corrupting the underlying application data.

The `TrafficManipulator` module integrates a sophisticated fragmentation engine that operates as a post-processing stage applied to already manipulated adversarial traffic.

## 2. System Architecture

The fragmentation logic is decoupled into a high-level orchestration layer and a low-level packet reconstruction layer.

### 2.1 Orchestration Layer (`manipulator.py`)
The `Manipulator` class serves as the controller. It manages the global configuration state and iterates through the packet stream. It does not perform the byte-level manipulation itself but decides *which* packets undergo the transformation based on stochastic parameters.

**Key Configuration Parameters:**
*   **$\rho$ (`fragment_prob`)**: The probability ($0.0 \le \rho \le 1.0$) that a candidate packet will be selected for fragmentation.
*   **$S_{min}$ (`min_fragment_size`)**: The lower bound for fragment payload size (default: 64 bytes).
*   **$S_{max}$ (`max_fragment_size`)**: The upper bound for fragment payload size (default: 1200 bytes).

### 2.2 Reconstruction Layer (`rebuilder.py`)
The `rebuilder` module implements the `fragment_packet` function, which handles the protocol-compliant dismantling of the IP packet. It ensures that the resulting stream of fragments is valid according to RFC 791.

## 3. Algorithmic Implementation

 The core algorithm transforms a set of packets $P = \{p_1, p_2, ..., p_n\}$ into a new set $P' = \{f_{1,1}, f_{1,2}, ..., f_{n,1}, ...\}$, where each $p_i$ may correspond to one or multiple fragments $f_{i,j}$.

### 3.1 Selection Process
For every packet $p_i$ in the stream:
1.  **Validity Check**: The packet must strictly contain an IPv4 layer (`pkt.haslayer(IP)`).
2.  **Size Constraint**: The payload size $L_{payload}$ must satisfy $L_{payload} > S_{min}$.
3.  **Stochastic Selection**: A random variable $r \sim U[0,1]$ is drawn. If $r \le \rho$, the packet enters the fragmentation routine.

### 3.2 Recursive Fragmentation Logic
The decomposition of the payload is performed iteratively. At each step, the algorithm determines the size of the next fragment $s_{frag}$ based on the remaining payload $L_{rem}$.

The maximum allowable size for the current fragment, $s_{limit}$, is calculated to ensure that if a remainder exists, it is large enough to form a valid final fragment:

$$
s_{limit} = \min(S_{max}, L_{rem} - S_{min})
$$

The specific size $s_{frag}$ is then drawn from a uniform distribution:

$$
s_{frag} \sim \mathcal{U}(S_{min}, s_{limit})
$$

### 3.3 Protocol Compliance and Constraints
To ensuring the fragmented traffic traverses network stacks (routers, firewalls, OS kernels) without being dropped, the implementation adheres to strict IP protocol constraints.

#### The 8-Byte Alignment Rule
The IP header `Fragment Offset` field is 13 bits wide and counts offset in units of 8 bytes (64 bits). Consequently, the length of the data payload for every fragment—**except the last one**—must be a multiple of 8.

The algorithm enforces this via truncation:
```python
if more_fragments and fragment_payload_size % 8 != 0:
    fragment_payload_size = (fragment_payload_size // 8) * 8
```

#### IP Header Field Management
For each fragment $f_j$ generated from packet $p$:
1.  **Identification (`IP.id`)**: A unique 16-bit ID is generated ($ID \sim \mathcal{U}[1, 65535]$) for the parent packet $p$ and assigned to all its constituent fragments $f_j$. This allows the receiver to reassemble them.
2.  **Flags (`IP.flags`)**:
    *   The `MF` (More Fragments) bit is set to 1 for all fragments $j < k$ (where $k$ is the substantial number of fragments).
    *   The `MF` bit is set to 0 for the final fragment $j=k$.
3.  **Offset (`IP.frag`)**: Calculated as the cumulative payload bytes sent thus far, divided by 8.
4.  **Metadata Preservation**: Critical fields (`src`, `dst`, `proto`, `ttl`, `tos`) are copied verbatim from $p$ to ensure proper routing.

## 4. Code Implementation Analysis

Below are the essential components extracted from the codebase illustrating the logic described above.

### 4.1 The Entry Point (`manipulator.py`)
This method iterates the packet list and applies the probability filter.

```python
def _apply_fragmentation(self, packets):
    """Apply random fragmentation to packets with configurable probability"""
    from rebuilder import fragment_packet
    fragmented_packets = []
    
    for i, pkt in enumerate(packets):
        if pkt.haslayer(IP):
            # Calculate actual payload size
            ip_layer = pkt[IP]
            ip_header_len = 20 if ip_layer.ihl is None else ip_layer.ihl * 4
            payload_size = len(pkt) - ip_header_len
            
            # Check size and probability constraints
            if payload_size > self.min_fragment_size:
                if np.random.random() <= self.fragment_prob:
                    # Logic delegates to the rebuilder
                    fragments = fragment_packet(
                        pkt, 
                        self.min_fragment_size, 
                        self.max_fragment_size
                    )
                    fragmented_packets.extend(fragments)
                else:
                    fragmented_packets.append(pkt)
            else:
                fragmented_packets.append(pkt)
        else:
            fragmented_packets.append(pkt)
            
    return fragmented_packets
```

### 4.2 The Core Logic (`rebuilder.py`)
This function handles the byte-level slicing and header construction.

```python
def fragment_packet(pkt, min_fragment_size=64, max_fragment_size=1200):
    # ... setup code omitted ...

    payload = bytes(pkt)[ip_header_len:]
    fragment_id = random.randint(1, 65535)  # Random ID for this group
    payload_offset = 0
    
    while payload_offset < len(payload):
        remaining_payload = len(payload) - payload_offset
        
        # Determine if we need another fragment after this one
        can_create_another = remaining_payload > (2 * min_fragment_size)
        
        if not can_create_another:
            # Final fragment takes everything
            fragment_payload_size = remaining_payload
            more_fragments = False
        else:
            # Randomize size for intermediate fragments
            max_allowed_size = min(max_fragment_size, 
                                 remaining_payload - min_fragment_size)
            
            fragment_payload_size = random.randint(min_fragment_size, max_allowed_size)
            more_fragments = True
        
        # CRITICAL: 8-Byte Alignment Enforcement
        if more_fragments and fragment_payload_size % 8 != 0:
            fragment_payload_size = (fragment_payload_size // 8) * 8
        
        # Construct Fragment ID
        fragment_pkt = IP()
        # ... copy flags ...
        fragment_pkt.id = fragment_id
        fragment_pkt.flags = 'MF' if more_fragments else 0
        fragment_pkt.frag = payload_offset // 8  # 8-byte units
        
        # ... append payload and L2 headers ...
        
        fragments.append(fragment_pkt)
        payload_offset += fragment_payload_size
        
    return fragments
```

## 5. Summary of Stochastic Properties

The strength of this evasion technique lies in its non-determinism. Two identical runs of the algorithm on the same Pcap file will yield different network footprints:

1.  **Selection Entropy**: A packet might be fragmented in run A but not in run B (governed by $\rho$).
2.  **Structural Entropy**: Even if a packet is fragmented in both runs, the *number* of fragments and their specific *sizes* will differ (governed by $s_{frag} \sim \mathcal{U}$).
3.  **Identifier Entropy**: The IP ID field is randomized, altering the header checksums and pattern signatures.

This variability forces ML models to generalize extremely well to fragmented traffic or risk classification failure due to the distribution shift in feature vectors (e.g., packet count, mean packet size).
