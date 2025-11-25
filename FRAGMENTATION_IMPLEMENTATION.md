# TrafficManipulator Fragmentation Enhancement

## Overview
Enhanced the TrafficManipulator with intelligent packet fragmentation capabilities. The fragmentation occurs **after** the PSO manipulation process, allowing for evasion techniques that combine traffic manipulation with packet fragmentation.

## Key Features Implemented

### 1. Configurable Fragmentation Probability
- Set fragmentation probability from 0% to 100%  
- Default: 100% (fragment all suitable packets)
- Use `--fragment_prob 0.5` for 50% fragmentation rate

### 2. Random Fragment Sizes  
- Fragments have randomized sizes within user-defined bounds
- Default range: 64-1200 bytes
- Minimum fragment size prevents overly small fragments
- Maximum fragment size controls fragmentation granularity

### 3. RFC-Compliant IP Fragmentation
- Proper IP fragmentation headers (ID, flags, fragment offset)
- 8-byte alignment for fragment offsets (required by RFC)
- More Fragments (MF) flag handling
- Automatic checksum recalculation

### 4. Intelligent Packet Selection
- Only fragments IP packets larger than minimum size
- Preserves non-IP packets and small packets unchanged
- Maintains original packet timing and metadata

## Command Line Usage

### Basic Fragmentation (100% probability)
```bash
python3 main.py -m malicious.pcap -b mimic_set.npy -n normalizer.pkl --fragment
```

### Custom Fragmentation Settings
```bash
# Fragment 70% of packets with sizes between 100-800 bytes
python3 main.py -m malicious.pcap -b mimic_set.npy -n normalizer.pkl \
  --fragment --fragment_prob 0.7 --min_fragment_size 100 --max_fragment_size 800

# Fragment 50% of packets with smaller fragments (64-400 bytes)  
python3 main.py -m malicious.pcap -b mimic_set.npy -n normalizer.pkl \
  --fragment --fragment_prob 0.5 --min_fragment_size 64 --max_fragment_size 400
```

## Technical Implementation

### Files Modified

#### 1. `main.py`
- Added `--fragment` flag to enable fragmentation
- Added `--fragment_prob` parameter (0.0-1.0)
- Added `--min_fragment_size` parameter (default: 64 bytes)
- Added `--max_fragment_size` parameter (default: 1200 bytes)

#### 2. `manipulator.py`  
- Added fragmentation parameters to Manipulator class
- Added `set_fragmentation_params()` method
- Added `_apply_fragmentation()` method with detailed statistics
- Integrated fragmentation into the processing pipeline (post-PSO)

#### 3. `rebuilder.py`
- Added `fragment_packet()` function with intelligent randomization
- Implemented proper IP header handling and fragment creation
- Added 8-byte alignment for fragment offsets  
- Preserved original packet metadata and timing

### Fragmentation Algorithm

1. **Packet Selection**: Only IP packets larger than `min_fragment_size`
2. **Probability Check**: Apply fragmentation based on `fragment_prob`
3. **Size Calculation**: Random fragment sizes within bounds
4. **Alignment**: Ensure 8-byte alignment (except last fragment)
5. **Header Creation**: Proper IP fragmentation headers
6. **Metadata Preservation**: Maintain timing and lower-layer headers

## Usage Statistics

The system provides detailed fragmentation statistics:

```
@Manipulator: Fragmentation Statistics:
  Total packets: 1000
  Fragmentable packets: 850
  Packets actually fragmented: 595  
  Total fragments created: 2380
  Final packet count: 2785
  Fragmentation ratio: 2.38x
```

## Evasion Benefits

### 1. Traffic Pattern Obfuscation
- Changes packet size distribution
- Alters inter-packet timing patterns  
- Creates multiple fragments from single packets

### 2. Detection Evasion
- Bypasses size-based detection rules
- Complicates flow reassembly for analyzers
- Introduces additional entropy in traffic

### 3. Randomization
- Fragment sizes are randomized within bounds
- Fragmentation probability adds unpredictability
- Each run produces different fragmentation patterns

## Testing

Comprehensive testing shows:
- ✅ Proper IP fragmentation header creation
- ✅ Correct fragment payload reconstruction  
- ✅ Preservation of packet metadata
- ✅ Configurable probability control
- ✅ Random fragment size generation
- ✅ 8-byte alignment compliance

## Example Output

```
Processing 100 packets with fragmentation probability 0.7
Fragment size range: [64, 800] bytes
Fragmenting packet 42/100 (size: 1500 bytes)
  Fragment 1: offset=0, size=320, MF=Yes
  Fragment 2: offset=320, size=480, MF=Yes  
  Fragment 3: offset=800, size=700, MF=No
Created 3 fragments from original packet
```

## Compatibility

- ✅ Compatible with all existing TrafficManipulator features
- ✅ Works with PSO optimization and particle manipulation  
- ✅ Preserves existing command-line interface
- ✅ Backward compatible (fragmentation is opt-in)

## Performance Impact

- Minimal CPU overhead for fragmentation logic
- Memory usage scales with number of fragments created
- I/O impact from larger output files (more packets)
- Network impact from increased packet count

---

**The enhanced TrafficManipulator now provides sophisticated packet fragmentation capabilities for advanced evasion scenarios while maintaining full compatibility with existing functionality.**
