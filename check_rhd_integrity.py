
import sys
import os
import glob
import numpy as np
import spikeinterface.extractors as se
from neo.rawio import IntanRawIO
import traceback

def check_file(file_path):
    print(f"Checking file: {file_path}")
    try:
        # Try reading with neo directly first to check header/data size
        print("  Attempting to read with neo.rawio.IntanRawIO...")
        io = IntanRawIO(filename=file_path)
        io.parse_header()
        print("  Header parsed successfully.")
        print(f"  Signal streams: {io.header['signal_streams']}")
        
        # Check data size manually if possible (simulating what might go wrong)
        # The error "Size of available data is not a multiple of the data-type size"
        # usually happens in np.memmap or similar when the file size doesn't match the expected structure.
        
        # Try reading with spikeinterface
        print("  Attempting to read with spikeinterface.extractors.read_intan...")
        rec = se.read_intan(file_path, stream_id="0") # Assuming stream_id 0, or let it detect
        print(f"  SpikeInterface read successfully. Duration: {rec.get_total_duration()}s")
        print("  File appears valid.")
        return True
    except Exception as e:
        print(f"  FAILED to read file: {file_path}")
        print(f"  Error: {e}")
        traceback.print_exc()
        return False

def main():
    if len(sys.argv) < 2:
        print("Usage: python check_rhd_integrity.py <path_to_file_or_folder> [more_paths ...]")
        sys.exit(1)

    failures = []
    
    for target_path in sys.argv[1:]:
        if os.path.isfile(target_path):
            if not check_file(target_path):
                failures.append(target_path)
        elif os.path.isdir(target_path):
            print(f"Scanning directory: {target_path}")
            rhd_files = glob.glob(os.path.join(target_path, "**/*.rhd"), recursive=True)
            print(f"Found {len(rhd_files)} .rhd files.")
            
            for rhd_file in rhd_files:
                if not check_file(rhd_file):
                    failures.append(rhd_file)
        else:
            print(f"Error: Path not found: {target_path}")
            failures.append(target_path)

    if failures:
        print(f"\nSummary: {len(failures)} files failed.")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print("\nAll checked files appear valid.")

if __name__ == "__main__":
    main()
