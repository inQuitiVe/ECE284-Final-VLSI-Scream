module core #(
  parameter bw      = 4,   // bit-width for weights/activations
  parameter psum_bw = 16,  // bit-width for partial sums
  parameter col     = 8,
  parameter row     = 8
)(
  input  clk,
  input  reset,

  // high level instructions from TB
  input [2:0] inst_w,

  // L0 mem ctrls from TB
  input              CEN_xmem,
  input              WEN_xmem,
  input [10:0]       A_xmem,
  input [bw*col-1:0] D_xmem,  // 32b
  input              CEN_wmem,
  input              WEN_wmem,
  input [7:0]       A_wmem,
  input [2*row*bw-1:0] D_wmem,  // 32b
  input              is_os,
  input              act_2b_mode,
  // // PSUM mem ctrls from TB
  // input           CEN_pmem,
  // input           WEN_pmem,
  // input   [10:0]  A_pmem,

  // from/to TB
  input  [3:0]            kij,  // for SFU
  input                   readout_start,  // trigger for output stage
  output [psum_bw*col-1:0] readout        // 16*8b
);

  // Wires
  wire [row*bw-1:0] L0fifo_data_in;  // 32b
  wire [2*row*bw-1:0] ififo_weight_in;  // 32b


  // PSUM mem
  wire [psum_bw*col-1:0] Q_pmem;  // 128b
  wire                   ren_pmem;
  wire                   wen_pmem;
  wire [3:0]             w_A_pmem;  // w=16
  wire [3:0]             r_A_pmem;  // w=16
  wire [psum_bw*col-1:0] D_pmem;  // 128b

  reg [2:0] inst_w_D1;  // due to memory's latency, inst_w has to be delayed 1 cycle before passing into corelet.v
  always @ (posedge clk) begin
    inst_w_D1 <= inst_w;
  end






  // Output Logic
  assign sfp_out = D_pmem;

  sram_32b_w2048 X_MEM_instance (
    // given by core_tb
    .CLK (clk                ),
    .D   (D_xmem             ),
    .CEN (CEN_xmem           ),
    .WEN (WEN_xmem           ),
    .A   (A_xmem             ),
    // connect to corelet vector_in
    .Q   (L0fifo_data_in     )  // 32 bit
  );

  sram_64b_w256 W_MEM_instance (
    // given by core_tb
    .CLK (clk                ),
    .D   (D_wmem             ),
    .CEN (CEN_wmem           ),
    .WEN (WEN_wmem           ),
    .A   (A_wmem             ),
    // connect to corelet vector_in
    .Q   (ififo_weight_in     )  // 32 bit
  );

  sram_128b_w16_RW P_MEM_instance (
    .CLK  (clk                ),
    .ren  (ren_pmem           ),
    .wen  (wen_pmem           ),
    .w_A  (w_A_pmem           ),
    .r_A  (r_A_pmem           ),
    .D    (D_pmem             ),
    .Q    (Q_pmem             )
  );

  corelet #(.bw(bw), .psum_bw(psum_bw), .col(col), .row(row)) corelet_instance (
    .clk          (clk                ),
    .reset        (reset              ),
    .inst_w       (inst_w_D1          ),
    .vector_data_in(L0fifo_data_in    ),
    .weight_in    (ififo_weight_in          ),
    // data in and ctrl signal for PSUM SRAM
    .Q_pmem       (Q_pmem             ),
    .ren_pmem     (ren_pmem           ),
    .wen_pmem     (wen_pmem           ),
    .w_A_pmem     (w_A_pmem           ),
    .r_A_pmem     (r_A_pmem           ),
    .D_pmem       (D_pmem             ),
    .is_os        (is_os              ),
    .act_2b_mode  (act_2b_mode         ),
    // for SFU
    .kij          (kij                ),
    .readout_start(readout_start      ),
    .readout      (readout            )  // 16*8b
  );


endmodule

