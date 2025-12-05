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
    > Note: the output.txt here is the output of conv layer. In the final delivery, it should be ReLU's output.
### software
- ipynb: tb file that I generated. It has the correct format nad structure, but I used my own model that has only 90% validation accuracy.
