import re

def find_lowest_mad(file_path):
    mad_entries = []
    current_epoch = None

    with open(file_path, 'r') as file:
        content = file.readlines()

        epoch_pattern = re.compile(r"Epoch \[(\d+)/\d+\]")
        mad_pattern = re.compile(r"- MAD: ([\d.]+)")

        for line_num, line in enumerate(content, 1):
            epoch_match = epoch_pattern.search(line)
            if epoch_match:
                current_epoch = int(epoch_match.group(1))
                
            match = mad_pattern.search(line)
            if match and current_epoch is not None:
                mad_value = float(match.group(1))
                mad_entries.append((mad_value, line_num, line.strip(), current_epoch))

    lowest_mad_entries = sorted(mad_entries, key=lambda x: x[0])[:5]
    
    return lowest_mad_entries

if __name__ == "__main__":
    file_path = r"D:\NCKHSV.2024-2025\H2F-GCN-Rehab\load.txt"
    lowest_mad = find_lowest_mad(file_path)
    
    print("Five lowest MAD values:")
    for value, line_num, line_text, epoch in lowest_mad:
        print(f"Epoch [{epoch}/1500]: MAD: {value}: {line_text}")