#!/usr/bin/env python3
"""Script to identify lines with encoding issues in prompts.txt"""

import sys

def check_file_encoding(filepath):
    """Check each line of the file for encoding issues."""
    try:
        with open(filepath, 'rb') as f:
            lines = f.readlines()
        
        print(f"Checking {len(lines)} lines in {filepath}\n")
        
        for line_num, line_bytes in enumerate(lines, start=1):
            try:
                # Try to decode as ASCII
                line_bytes.decode('ascii')
            except UnicodeDecodeError as e:
                print(f"❌ ERROR on line {line_num}:")
                print(f"   Error: {e}")
                print(f"   Problematic byte: 0x{e.object[e.start]:02x} at position {e.start}")
                
                # Show the raw bytes around the error
                start = max(0, e.start - 10)
                end = min(len(line_bytes), e.end + 10)
                problematic_bytes = line_bytes[start:end]
                print(f"   Bytes around error: {problematic_bytes}")
                print(f"   Hex: {' '.join(f'{b:02x}' for b in problematic_bytes)}")
                
                # Try to decode as UTF-8 to show what it should be
                try:
                    utf8_line = line_bytes.decode('utf-8')
                    print(f"   Decoded as UTF-8: {repr(utf8_line[:100])}...")
                    # Show the problematic character
                    if e.start < len(utf8_line):
                        char = utf8_line[e.start]
                        print(f"   Problematic character: {repr(char)} (Unicode: U+{ord(char):04X})")
                except:
                    pass
                
                print()
        
        print("✅ Check complete!")
        
    except FileNotFoundError:
        print(f"Error: File '{filepath}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    filepath = "prompts.txt"
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    
    check_file_encoding(filepath)



