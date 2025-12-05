# README.md
## Summary & Outline
For project delivery.\
Part 1. Vanilla Version\
Part 2. 2b/4b reconfigurable SIMD Version\
Part 3. OS/WS reconfigurable Version\
Part 4. Fancy Alpha Version

## Structure
### hardware
- src: verilog code (contains tb)
- golden: txt files that tb needs, and the ones with "viz" prefix are for human readibility.
    > Note: the out_raw.txt here is the output of conv layer. out.txt is ReLU's output.
    The tb is using out.txt now. 
### software
- ipynb: golden pattern generator.