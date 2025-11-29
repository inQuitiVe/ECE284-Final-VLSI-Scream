module core #(
  parameter bw       = 4,    // bit-width for weights/activations
  parameter psum_bw  = 16,   // bit-width for partial sums
  parameter col      = 8,
  parameter row      = 8
)(
    input           clk,
    input           reset,

    // high level instructions from TB
    input   [1:0]   inst_w,

    // L0 mem ctrls from TB
    input           CEN_xmem,
    input           WEN_xmem,
    input   [10:0]  A_xmem,
    input   [bw*col-1:0] D_xmem, //32b

    // PSUM mem ctrls from TB
    input           CEN_pmem,
    input           WEN_pmem,
    input   [10:0]  A_pmem,

    // Output to TB
    output          valid,
    output  [psum_bw*col-1:0] sfp_out //16*8b
);

// Wires
wire [row*bw-1:0] L0fifo_data_in;  //32b

reg [1:0]  inst_w_D1; //due to memory's latency, inst_w has to be delayed 1 cycle before passing into corelet.v 
always @(posedge clk ) begin
  inst_w_D1 <= inst_w;
end






// Output Logic
assign valid = 1'b0;
assign sfp_out = {(psum_bw*col){1'b0}};

sram_32b_w2048 X_MEM_instance(
  // given by core_tb
  .CLK(clk), 
  .D(D_xmem), 
  .CEN(CEN_xmem), 
  .WEN(WEN_xmem), 
  .A(A_xmem),
  // connect to corelet vector_in
  .Q(L0fifo_data_in)  //32 bit
);

sram_32b_w2048 P_MEM_instance(
  .CLK(clk),
  .CEN(), 
  .WEN(), 
  .A(),
  .D(), 
  .Q()
);

corelet #(.bw(bw), .psum_bw(psum_bw), .col(col), .row(row)) corelet_instance(
  .clk(clk),
  .reset(reset),
  .inst_w(inst_w_D1),
  .vector_data_in(L0fifo_data_in)
);



endmodule