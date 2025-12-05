#!/usr/bin/env python3
"""
Generate expected_output_from_psum.txt from calc_psum_output.txt

- Uses the same accumulation rule as SFU (nij_cnt, kij -> o_addr)
- For each o_addr (0~15), accumulates psums over all kij and nij_cnt
- Applies ReLU (since SFU has ReLU on Q_pmem before readout)
- Writes file in the same human-readable format as viz_output.txt:
  - 3 header lines
  - 16 data lines, each with 16 columns
    * First 8 columns: accumulated psum (after ReLU) for 8 MAC columns
    * Last 8 columns: filled with 0 (we only computed 8 channels)
"""

import os
import re


def relu(x: int) -> int:
  return x if x > 0 else 0


def calculate_o_addr(nij_cnt: int, kij: int):
  """
  Match SFU.v logic:
    row_a = nij_cnt / 6
    col_a = nij_cnt % 6
    k_row = kij / 3
    k_col = kij % 3
    o_row = row_a - k_row
    o_col = col_a - k_col
    valid if 0 <= o_row < 4 and 0 <= o_col < 4
    o_addr = o_row * 4 + o_col
  """
  row_a = nij_cnt // 6
  col_a = nij_cnt % 6
  k_row = kij // 3
  k_col = kij % 3

  o_row = row_a - k_row
  o_col = col_a - k_col

  if 0 <= o_row < 4 and 0 <= o_col < 4:
    return o_row * 4 + o_col, True
  return 0, False


def read_calc_psum_output(path: str):
  """
  Read calc_psum_output.txt
  Returns dict[kij][nij_cnt] = [psum0..psum7]
  """
  psums_by_kij_nij = {}

  with open(path, "r") as f:
    lines = f.readlines()

  for line in lines:
    line = line.strip()
    if not line or line.startswith("#"):
      continue

    # Lines look like: kij= 0 time= 7:  2  4 ...
    m = re.match(r"kij\s*=\s*(\d+)\s+time\s*=\s*(\d+):\s+(.+)", line)
    if not m:
      continue

    kij = int(m.group(1))
    nij_cnt = int(m.group(2))
    psum_str = m.group(3)
    vals = [int(x) for x in psum_str.split()]

    if kij not in psums_by_kij_nij:
      psums_by_kij_nij[kij] = {}
    psums_by_kij_nij[kij][nij_cnt] = vals

  return psums_by_kij_nij


def accumulate_by_o_addr(psums_by_kij_nij, num_kij=9, num_nij=36):
  """
  Accumulate psums by o_addr (0~15).
  Returns dict[o_addr] = [psum0..psum7]
  """
  o_addr_psums = {o: [0] * 8 for o in range(16)}

  for kij in range(num_kij):
    if kij not in psums_by_kij_nij:
      continue
    for nij_cnt in range(num_nij):
      if nij_cnt not in psums_by_kij_nij[kij]:
        continue
      o_addr, acc = calculate_o_addr(nij_cnt, kij)
      if not acc:
        continue
      vec = psums_by_kij_nij[kij][nij_cnt]
      for c in range(8):
        o_addr_psums[o_addr][c] += vec[c]

  return o_addr_psums


def main():
  script_dir = os.path.dirname(os.path.abspath(__file__))
  calc_path = os.path.join(script_dir, "calc_psum_output.txt")
  out_path = os.path.join(script_dir, "expected_output_from_psum.txt")

  if not os.path.exists(calc_path):
    print(f"[FAIL] calc_psum_output.txt not found at {calc_path}")
    return

  print("[1] Reading calc_psum_output.txt ...")
  psums_by_kij_nij = read_calc_psum_output(calc_path)
  print(f"    Loaded psums for {len(psums_by_kij_nij)} kij rounds")

  print("[2] Accumulating by o_addr ...")
  o_addr_psums = accumulate_by_o_addr(psums_by_kij_nij)

  print("[3] Applying ReLU ...")
  for o in range(16):
    o_addr_psums[o] = [relu(v) for v in o_addr_psums[o]]

  print("[4] Writing expected_output_from_psum.txt (viz format) ...")
  lines_viz = []
  # Header copied from viz_output style
  lines_viz.append("#time0row7[msb-lsb],time0row6[msb-lst],....,time0row0[msb-lst]#")
  lines_viz.append("#time1row7[msb-lsb],time1row6[msb-lst],....,time1row0[msb-lst]#")
  lines_viz.append("#................#")

  # Each o_addr is one row (0..15)
  for o in range(16):
    vals8 = o_addr_psums[o]
    # Last 8 columns are zero (not computed)
    vals16 = vals8[::-1]
    line = "".join(f"{v:7d}" for v in vals16)
    lines_viz.append(line)

  with open(out_path, "w") as f:
    f.write("\n".join(lines_viz) + "\n")

  print("[OK] expected_output_from_psum.txt generated.")
  print(f"     Output : {out_path}")

  print("[5] Writing expected_output_from_psum_binary.txt (binary format) ...")
  out_binary_path = os.path.join(script_dir, "expected_output_from_psum_binary.txt")
  lines_binary = []
  # Header copied from output.txt style
  lines_binary.append("#time0row7[msb-lsb],time0row6[msb-lst],....,time0row0[msb-lst]#")
  lines_binary.append("#time1row7[msb-lsb],time1row6[msb-lst],....,time1row0[msb-lst]#")
  lines_binary.append("#................#")

  # Each o_addr is one row (0..15)
  for o in range(16):
    vals8 = o_addr_psums[o]
    # Last 8 columns are zero (not computed)
    # Reverse the first 8 values, then pad with 8 zeros to make 16 channels
    vals16 = vals8[::-1]
    # Convert each value to 16-bit signed binary
    # Handle negative values using 2's complement
    binary_str = ""
    for v in vals16:
      # Convert to 16-bit signed integer (2's complement)
      if v < 0:
        v_16bit = (1 << 16) + v  # 2's complement for negative
      else:
        v_16bit = v
      # Format as 16-bit binary string
      binary_str += format(v_16bit & 0xFFFF, "016b")
    lines_binary.append(binary_str)

  with open(out_binary_path, "w") as f:
    f.write("\n".join(lines_binary) + "\n")

  print("[OK] expected_output_from_psum_binary.txt generated.")
  print(f"     Output : {out_binary_path}")


if __name__ == "__main__":
  main()


