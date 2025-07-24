import pandas as pd
import numpy as np
import joblib
import sys
import time
from scapy.all import sniff, IP, TCP, UDP, Raw # Import necessary Scapy components

# --- 2.1 Load Trained Model, Threshold, and Fitted Preprocessor ---
model_load_path = 'isolation_forest_model.joblib'
threshold_load_path = 'anomaly_threshold.txt'
preprocessor_load_path = 'fitted_preprocessor.joblib'

try:
    model = joblib.load(model_load_path)
    with open(threshold_load_path, 'r') as f:
        original_threshold = float(f.read()) # Load the original threshold value

    # >>>>>>> MANUAL ADJUSTMENT FOR TESTING ANOMALY DETECTION <<<<<<<
    # Adjust this value to make the model more sensitive.
    # Lower (more negative) values mean more packets will be flagged as anomalies.
    # Try 0.00, -0.01, -0.02, etc., if you want to see more alerts.
    chosen_threshold = 0.00 # Set a very sensitive threshold for demonstration
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    preprocessor = joblib.load(preprocessor_load_path)

    print("Real-time Monitor: All assets (model, preprocessor) loaded successfully.")
    print(f"Original Anomaly Threshold (from file): {original_threshold:.4f}")
    print(f"Adjusted Anomaly Threshold (for testing): {chosen_threshold:.4f}")

except FileNotFoundError as e:
    print(f"Error loading assets: {e}. Ensure model_trainer.py and preprocessing.ipynb ran correctly and files exist.")
    sys.exit(1) # Exit if essential files are missing
except Exception as e:
    print(f"An unexpected error occurred during asset loading: {e}")
    sys.exit(1)

# --- 2.2 Define Constants for Feature Engineering (MUST MATCH preprocessing.ipynb) ---
# These are crucial for creating features from live packets in a consistent way.
common_ports = {
    'port_21_ftp': 21, 'port_22_ssh': 22, 'port_23_telnet': 23, 'port_80_http': 80,
    'port_443_https': 443, 'port_3389_rdp': 3389, 'port_8080_proxy': 8080
}
protocol_map = {1: 'ICMP', 6: 'TCP', 17: 'UDP'}

# Columns that were dropped *before* the ColumnTransformer in preprocessing.ipynb
# This list helps in creating the correct input DataFrame structure for the preprocessor.
# 'minute_of_hour' has been permanently removed from this list as per previous fixes.
features_to_drop_for_preprocessing = [
    'timestamp', 'src_ip', 'dst_ip', 'flow_key', 'bidirectional_flow_key',
    'sorted_ips', 'sorted_ports', 'protocol' # 'minute_of_hour' should NOT be here
]

# Get the exact list of numerical features the preprocessor was trained on
# This is retrieved from the loaded preprocessor object
try:
    numerical_cols_from_preprocessor = preprocessor.named_transformers_['num'].get_feature_names_out()
except Exception as e:
    print(f"Warning: Could not get numerical column names from preprocessor. Defaulting to empty list. Error: {e}")
    numerical_cols_from_preprocessor = [] # Fallback
#### **Step 3: Define Packet Parsing and Feature Engineering Function for Live Traffic**

#### **Step 3: Define Packet Parsing and Feature Engineering Function for Live Traffic**

def parse_live_packet_to_df_row(packet):
    """
    Parses a single Scapy packet and extracts features into a DataFrame row.
    This function must replicate the feature engineering logic from preprocessing.ipynb,
    but adapted for a single packet and filling flow-based features with defaults.
    """
    row_data = {}

    if IP in packet: # Check if the packet has an IP layer
        # Packet-level raw features
        row_data['timestamp'] = packet.time
        row_data['src_ip'] = str(packet[IP].src)
        row_data['dst_ip'] = str(packet[IP].dst)
        row_data['protocol'] = packet[IP].proto
        row_data['packet_length'] = len(packet)

        # Port information
        src_port = None
        dst_port = None
        if TCP in packet:
            src_port = packet[TCP].sport
            dst_port = packet[TCP].dport
        elif UDP in packet:
            src_port = packet[UDP].sport
            dst_port = packet[UDP].dport
        row_data['src_port'] = src_port
        row_data['dst_port'] = dst_port

        # Time-based features (from single packet's timestamp)
        dt_object = pd.to_datetime(row_data['timestamp'], unit='s', errors='coerce')
        if pd.notna(dt_object):
            row_data['hour_of_day'] = dt_object.hour
            row_data['day_of_week'] = dt_object.dayofweek
            row_data['minute_of_hour'] = dt_object.minute # Ensure this matches preprocessing.ipynb
        else:
            row_data['hour_of_day'] = 0
            row_data['day_of_week'] = 0
            row_data['minute_of_hour'] = 0

        # Port presence indicators (must match common_ports definition)
        for feature_name, port_num in common_ports.items():
            row_data[feature_name] = int((row_data['src_port'] == port_num) | (row_data['dst_port'] == port_num))

        # Protocol name
        row_data['protocol_name'] = protocol_map.get(row_data['protocol'], str(row_data['protocol']))

        # --- Handling Flow-Based Features (CRITICAL COMPROMISE FOR REAL-TIME PER-PACKET) ---
        # These features cannot be computed from a single packet without stateful tracking.
        # For this demonstration, we fill them with zeros (or a consistent default).
        # In a production system, you'd need stateful flow aggregation.
        row_data['bidir_flow_duration'] = 0.0
        row_data['bidir_total_packets'] = 1.0 # This packet is 1 packet in its (conceptual) flow
        row_data['bidir_total_bytes'] = row_data['packet_length'] # This packet's length is its flow's bytes
        row_data['bidir_mean_packet_length'] = row_data['packet_length']
        row_data['bidir_std_packet_length'] = 0.0 # Single packet, so std dev is 0
        row_data['num_unique_src_ips'] = 1.0
        row_data['num_unique_dst_ips'] = 1.0

        # Add temporary/intermediate columns that will be dropped later,
        # but needed for consistency with features_to_drop_for_preprocessing list in Cell 2 of eval.ipynb.
        # Ensure these are handled consistently or removed from features_to_drop_for_preprocessing if not needed.
        # For simplicity, we create dummy versions here if they are in the drop list.
        # This prevents 'columns missing' errors from preprocessor.transform if a drop column never existed.
        row_data['flow_key'] = 'dummy_flow'
        row_data['bidirectional_flow_key'] = 'dummy_bidirectional_flow'
        row_data['sorted_ips'] = ('dummy_ip1', 'dummy_ip2')
        row_data['sorted_ports'] = ('dummy_port1', 'dummy_port2')

        # Convert ports to string for OneHotEncoder (consistent with preprocessing.ipynb)
        row_data['src_port'] = str(int(row_data['src_port'])) if row_data['src_port'] is not None else '-1'
        row_data['dst_port'] = str(int(row_data['dst_port'])) if row_data['dst_port'] is not None else '-1'

        return pd.DataFrame([row_data]) # Return as DataFrame with one row
    else:
        return pd.DataFrame() # Return empty DataFrame if not an IP packet
def process_packet_callback(packet):
    """
    Callback function to process each sniffed packet for anomaly detection.
    """
    if IP in packet: # Only process IP packets
        start_time = time.time()
        try:
            # 1. Feature Engineering for the single packet
            df_single_packet_engineered = parse_live_packet_to_df_row(packet)

            if df_single_packet_engineered.empty:
                # print("Skipping non-IP packet or unparseable packet.")
                return

            # Ensure all numerical columns expected by preprocessor exist and are filled
            # We explicitly check against `numerical_cols_from_preprocessor` which are the true features
            # that `preprocessor.named_transformers_['num']` expects.
            for col in numerical_cols_from_preprocessor:
                if col not in df_single_packet_engineered.columns:
                    # print(f"Warning: Missing numerical feature '{col}' in live packet. Filling with 0.")
                    df_single_packet_engineered[col] = 0.0
            
            # Ensure categorical features are string type and filled as expected by preprocessor
            # We are not dynamically adding new categorical columns (e.g. for new ports),
            # 'handle_unknown="ignore"' in OneHotEncoder deals with unseen categories.
            for col_name in ['protocol_name', 'src_port', 'dst_port']:
                if col_name not in df_single_packet_engineered.columns:
                    df_single_packet_engineered[col_name] = '-1' # Default for missing categorical
                else:
                    df_single_packet_engineered[col_name] = df_single_packet_engineered[col_name].astype(str)


            # Drop columns that were dropped during preprocessing training before transformation
            df_final_input_for_transform = df_single_packet_engineered.drop(
                columns=features_to_drop_for_preprocessing, errors='ignore'
            )

            # 2. Preprocess the features using the loaded preprocessor
            # Use .transform(), NOT .fit_transform(), as it uses the learned scaling/encoding.
            X_preprocessed_live = preprocessor.transform(df_final_input_for_transform)

            # 3. Get anomaly score from the model
            anomaly_score = model.decision_function(X_preprocessed_live)[0]

            # 4. Classify as anomaly or normal based on threshold
            is_anomaly = anomaly_score <= chosen_threshold

            processing_time = (time.time() - start_time) * 1000 # in ms

            # 5. Alerting
            if is_anomaly:
                alert_msg = f"!!! ANOMALY DETECTED !!! " \
                            f"Time: {pd.to_datetime(packet.time, unit='s').strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]} " \
                            f"Src: {packet[IP].src}:{row_data.get('src_port', 'N/A')} " \
                            f"Dst: {packet[IP].dst}:{row_data.get('dst_port', 'N/A')} " \
                            f"Proto: {row_data.get('protocol_name', 'N/A')} " \
                            f"Score: {anomaly_score:.4f} (Threshold: {chosen_threshold:.4f}) " \
                            f"[Proc Time: {processing_time:.2f}ms]"
                print(alert_msg)
            else:
                # Optional: print normal traffic for debugging/monitoring
                print(f"Normal traffic. Score: {anomaly_score:.4f} Src: {packet[IP].src} Dst: {packet[IP].dst}")
                pass # Suppress normal traffic prints for cleaner output

        except Exception as e:
            print(f"Error processing packet: {e} | Packet: {packet.summary()}")
            # print(f"Problematic df_final_input_for_transform columns: {df_final_input_for_transform.columns.tolist()}")
            # print(f"Problematic df_final_input_for_transform head: \n{df_final_input_for_transform.head()}")
            pass # Continue to next packet even if one fails

if __name__ == "__main__":
    # --- Prompt for Network Interface ---
    # You need to know your network interface name.
    # On Windows, it might be something like "Ethernet", "Wi-Fi", or a GUID like "{XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX}"
    # On Linux/macOS, it might be "eth0", "wlan0", "en0", "lo" (loopback)
    print("\n--- Starting Real-time Network Anomaly Monitoring ---")
    print("WARNING: Live packet capture requires root/administrator privileges.")
    print("If you get a 'Permission denied' error, try running this script with 'sudo python realtime_monitor.py' (Linux/macOS) or as Administrator (Windows).")
    print("Press Ctrl+C to stop the monitoring.")

    # List available interfaces (requires root/admin sometimes)
    # try:
    #     from scapy.all import show_interfaces
    #     show_interfaces()
    # except Exception as e:
    #     print(f"Could not list interfaces: {e}. Please manually provide your interface name.")

    interface = input("Enter your network interface name (e.g., 'Ethernet', 'Wi-Fi', 'eth0', 'en0'): ").strip()

    if not interface:
        print("No interface provided. Exiting.")
        sys.exit(1)

    print(f"\nMonitoring interface: {interface}")
    print("Capturing packets... (Ctrl+C to stop)")

    try:
        # Start sniffing packets
        # prn=process_packet_callback: Calls our function for each packet
        # store=0: Don't store packets in memory (save RAM)
        # filter="ip": Only capture IP packets (reduces noise)
        # timeout=60: Stop after 60 seconds (optional, for testing)
        sniff(iface=interface, prn=process_packet_callback, store=0, filter="ip")

    except PermissionError:
        print("\nPermission denied: You need root/administrator privileges to capture live packets.")
        print("Try running the script with 'sudo' (Linux/macOS) or 'Run as Administrator' (Windows).")
    except Exception as e:
        print(f"\nAn error occurred during sniffing: {e}")
    finally:
        print("\nMonitoring stopped.")