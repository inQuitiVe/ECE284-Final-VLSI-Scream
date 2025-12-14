// Created by prof. Mingu Kang @VVIP Lab in UCSD ECE department
// Please do not spread this code without permission 
module mac_array (clk, reset, out_s, in_w, in_n, inst_w, valid);

  parameter bw = 4;
  parameter psum_bw = 16;
  parameter col = 8;
  parameter row = 8;

  input  clk, reset;
  output [psum_bw*col-1:0] out_s;
  input  [row*bw-1:0] in_w; // inst[1]:execute, inst[0]: kernel loading
  input  [1:0] inst_w;
  input  [psum_bw*col-1:0] in_n;
  output [col-1:0] valid;

  wire [psum_bw*col*row-1:0] out_s_temp;
  wire [col*row-1:0] valid_temp;
  wire [psum_bw*col*(row+1)-1:0] psum_temp;

  assign psum_temp[psum_bw*col-1:0] = in_n;
  assign out_s = out_s_temp[psum_bw*col*row-1:psum_bw*col*(row-1)];
  assign valid = valid_temp[col*row-1:col*(row-1)];

  integer idx;
  reg [1:0] inst_sr [0:col-1];
  always@(*) begin
    inst_sr[0] = inst_w;
  end
  always@(posedge clk) begin
    if (reset) begin
      for (idx=1; idx< col; idx= idx+ 1) begin
        inst_sr[idx]    <=  0;
      end 
    end else begin
      for (idx = 1; idx < col; idx = idx+1) begin
        inst_sr[idx]    <= inst_sr[idx-1];
      end
    end
  end

  genvar i;
  generate
    for (i=0; i < row ; i=i+1) begin : row_num
        mac_row #(.bw(bw), .psum_bw(psum_bw), .col(col)) mac_row_instance (
          .clk(clk),
          .reset(reset),
          .out_s(out_s_temp[psum_bw*col*(i+1)-1:psum_bw*col*i]),
          .in_w(in_w[bw*(i+1)-1:bw*i]),
          .in_n(psum_temp[psum_bw*col*(i+1)-1:psum_bw*col*i]),
          .valid(valid_temp[col*(i+1)-1:col*i]),
          .inst_w(inst_sr[i]));
        assign psum_temp[psum_bw*col*(i+2)-1:psum_bw*col*(i+1)] = out_s_temp[psum_bw*col*(i+1)-1:psum_bw*col*i];
    end
  endgenerate


endmodule
