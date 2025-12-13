module corelet #(
  parameter bw      = 4,   // bit-width for weights/activations
  parameter psum_bw = 16,  // bit-width for partial sums
  parameter col     = 8,
  parameter row     = 8
)(
  input  clk,
  input  reset,

  // high level instructions from TB
  input [2:0]            inst_w,
  input [row*bw-1:0]    vector_data_in,
  input [2*row*bw-1:0]  weight_in,

  // PSUM mem
  input  [psum_bw*col-1:0] Q_pmem,
  output                  ren_pmem,
  output                  wen_pmem,
  output [3:0]            w_A_pmem,
  output [3:0]            r_A_pmem,
  output [psum_bw*col-1:0] D_pmem,
  input                   is_os,
  input                   act_2b_mode,

  // For SFU
  input  [3:0]            kij,
  input                   readout_start,  // trigger for output stage
  output [psum_bw*col-1:0] readout         // 16*8b
);


  // L0fifo ctrl signal
  reg [2:0] inst_w_D1, inst_w_D2;
  always @ (posedge clk) begin
    if (reset) begin
      inst_w_D1 <= 3'b000;
      inst_w_D2 <= 3'b000;
    end
    else begin
      inst_w_D1 <= inst_w;  // start reading out data after writing
      inst_w_D2 <= inst_w_D1;
    end
  end



  wire [row*bw-1:0]      mac_array_data_in;   // connect to l0fifo.out
  wire [col*psum_bw-1:0] mac_array_data_out;   // connect to ofifo.in
  wire [col*psum_bw-1:0] ofifo_data_out;      // connect to SFU
  wire [col-1:0]         ofifo_wr;            // connect to mac_array.valid
  wire                   ofifo_valid;        // self connected to ofifo.rd, i.e. no latency
  wire [2*row*bw-1:0] mac_array_weight_raw;

  wire [psum_bw*col-1:0]  mac_array_weight_in;

  genvar i;
  generate
    for (i = 0; i < row; i = i + 1) begin
      assign mac_array_weight_in[i*psum_bw +: psum_bw] = is_os ? {8'b0, mac_array_weight_raw[i*2*bw +: 2*bw]}: {(psum_bw){1'b0}};
    end
  endgenerate


  l0 #(.bw(bw), .row(row)) L0fifo_instance (
    .clk     (clk                     ),
    .reset   (reset                   ),
    .in      (vector_data_in          ),
    .out     (mac_array_data_in       ),
    .rd      (inst_w_D1[1]|inst_w_D1[0]              ),  // start reading out data after writing
    .wr      (inst_w[1]|inst_w[0]                ),  // either kernel_load or execution, capture data
    .o_full  (                        ),
    .o_ready (                        )
  );

  ififo #(.bw(2*bw), .row(row)) ififo_instance (
    .clk     (clk                     ),
    .reset   (reset                   ),
    .in      (weight_in          ),
    .out     (mac_array_weight_raw       ),
    .rd      (inst_w_D1[1]|inst_w_D1[0]              ),  // start reading out data after writing
    .wr      (inst_w[1]|inst_w[0]                ),  // either kernel_load or execution, capture data
    .o_full  (                        ),
    .o_ready (                        )
  );

  mac_array #(.bw(bw), .psum_bw(psum_bw), .col(col), .row(row)) mac_array_instance (
    .clk        (clk                     ),
    .reset      (reset                   ),
    .out_s      (mac_array_data_out       ),
    .in_w       (mac_array_data_in        ),
    .in_n       (mac_array_weight_in    ),
    .inst_w     (inst_w_D2               ),
    .is_os      (is_os                   ),
    .act_2b_mode(act_2b_mode             ),
    .valid      (ofifo_wr                )
  );


  ofifo #(.bw(psum_bw), .col(col)) ofifo_instance (
    .clk     (clk                     ),
    .reset   (reset                   ),
    .in      (mac_array_data_out       ),
    .out     (ofifo_data_out           ),
    .rd      (ofifo_valid             ),
    .wr      (ofifo_wr                ),
    .o_full  (                        ),
    .o_ready (                        ),
    .o_valid (ofifo_valid             )
  );

  SFU #(.psum_bw(psum_bw), .col(col)) SFU_instance (
    .clk          (clk                     ),
    .reset        (reset                   ),
    // sense signal from ofifo and output ctrl
    .ofifo_valid  (ofifo_valid             ),
    .ofifo_data   (ofifo_data_out           ),
    // data in and ctrl signal for PSUM SRAM
    .Q_pmem       (Q_pmem                  ),
    .ren_pmem     (ren_pmem                ),
    .wen_pmem     (wen_pmem                ),
    .w_A_pmem     (w_A_pmem                ),
    .r_A_pmem     (r_A_pmem                ),
    .D_pmem       (D_pmem                  ),
    // output, valid
    .kij          (kij                     ),
    .readout_start(readout_start           ),
    .readout      (readout                 )  // 16*8b
  );


endmodule


