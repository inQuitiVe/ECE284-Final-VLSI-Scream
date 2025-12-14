// Created by prof. Mingu Kang @VVIP Lab in UCSD ECE department
// Please do not spread this code without permission 
module ofifo (clk, in, out, rd, wr, o_full, reset, o_ready, o_valid);

  parameter col = 8;
  parameter bw  = 4;

  input  clk;
  input  [col-1:0]      wr;
  input                 rd;
  input                 reset;
  input  [bw*col-1:0]  in;
  output [bw*col-1:0]  out;
  output                o_full;
  output                o_ready;
  output                o_valid;

  wire [col-1:0] empty;
  wire [col-1:0] full;
  reg            rd_en;

  genvar i;

  assign o_full  = |full;   // any FIFO has full=1, which is OR
  assign o_ready = ~o_full; // all FIFO has full=0, which is NOR
  assign o_valid = ~|empty; // all FIFOs has empty=0, which is NOR

  // Using depth=8 causes error!
  generate
    for (i=0; i<col; i=i+1) begin : col_num
      fifo_depth64 #(.bw(bw)) fifo_instance (
        .rd_clk  (clk                    ),
        .wr_clk  (clk                    ),
        .rd      (rd                     ),
        .wr      (wr[i]                  ),
        .o_empty (empty[i]               ),
        .o_full  (full[i]                ),
        .in      (in[(i+1)*bw-1:i*bw]    ),
        .out     (out[(i+1)*bw-1:i*bw]   ),
        .reset   (reset                  )
      );
    end
  endgenerate


  always @ (posedge clk) begin
    if (reset) begin
      rd_en <= 0;
    end
    else begin
      rd_en <= rd;
    end
  end

  // dbg wire
  // wire [15:0] dbg_in0_ofifo = in[15:0];
  // wire [15:0] dbg_in1_ofifo = in[31:16];
  // wire [15:0] dbg_in2_ofifo = in[47:32];
  // wire [15:0] dbg_in3_ofifo = in[63:48];
  // wire [15:0] dbg_in4_ofifo = in[79:64];
  // wire [15:0] dbg_in5_ofifo = in[95:80];
  // wire [15:0] dbg_in6_ofifo = in[111:96];
  // wire [15:0] dbg_in7_ofifo = in[127:112];
  
  // wire [15:0] dbg_out0_ofifo = out[15:0];
  // wire [15:0] dbg_out1_ofifo = out[31:16];
  // wire [15:0] dbg_out2_ofifo = out[47:32];
  // wire [15:0] dbg_out3_ofifo = out[63:48];
  // wire [15:0] dbg_out4_ofifo = out[79:64];
  // wire [15:0] dbg_out5_ofifo = out[95:80];
  // wire [15:0] dbg_out6_ofifo = out[111:96];
  // wire [15:0] dbg_out7_ofifo = out[127:112];
 

endmodule
