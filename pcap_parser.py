from scapy.all import rdpcap, IP, TCP, UDP
import pandas as pd

def parse_pcap(pcap_file):
    packets = rdpcap(pcap_file)
    data = []
    for pkt in packets:
        if IP in pkt: # Check if the packet has an IP layer
            ip_src = pkt[IP].src
            ip_dst = pkt[IP].dst
            protocol = pkt[IP].proto
            pkt_len = len(pkt)
            timestamp = pkt.time # Get the timestamp of the packet

            # Initialize ports to None, in case the packet is not TCP or UDP
            src_port = None
            dst_port = None

            # Check for TCP layer
            if TCP in pkt:
                src_port = pkt[TCP].sport
                dst_port = pkt[TCP].dport
            # Check for UDP layer
            elif UDP in pkt:
                src_port = pkt[UDP].sport
                dst_port = pkt[UDP].dport

            # Add more features as needed (flags, timestamps, etc.)
            row = {
                'timestamp': timestamp,
                'src_ip': str(ip_src), # Convert IP addresses to string
                'dst_ip': str(ip_dst),
                'protocol': protocol,
                'src_port': src_port, # Will be None if not TCP/UDP
                'dst_port': dst_port, # Will be None if not TCP/UDP
                'packet_length': pkt_len
            }
            data.append(row)
    return pd.DataFrame(data)

if __name__ == "__main__":
    pcap_file_path = 'dns.cap'

    print(f"Parsing PCAP file: {pcap_file_path}...")
    df = parse_pcap(pcap_file_path)
    print("Done parsing.")

    if not df.empty:
        print("\nExtracted features head:")
        print(df.head())
        output_csv_path = 'extracted_features.csv'
        df.to_csv(output_csv_path, index=False)
        print(f"\nFeatures saved to {output_csv_path}")
    else:
        print("\nNo IP packets found in the PCAP or DataFrame is empty. Check your PCAP file or its path.")