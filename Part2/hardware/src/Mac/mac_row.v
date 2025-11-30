// Created by prof. Mingu Kang @VVIP Lab in UCSD ECE department
// Please do not spread this code without permission 
module mac_row (clk, out_s, in_w, in_n, valid, inst_w, reset, is_os, act_2b_mode);

  parameter bw = 4;
  parameter psum_bw = 16;
  parameter col = 8;

  input  clk, reset;
  output [psum_bw*col-1:0] out_s;
  output [col-1:0] valid;
  input  [bw-1:0] in_w; // inst[1]:execute, inst[0]: kernel loading
  input  [1:0] inst_w;
  input  [psum_bw*col-1:0] in_n;
  input  is_os;
  input  act_2b_mode;
  wire  [(col+1)*bw-1:0] temp;
  assign temp[bw-1:0]   = in_w;

  wire  [(col+1)*2-1:0]  inst_bus;
  assign inst_bus[1:0]   = inst_w;

  genvar i;
  generate
    for (i=1; i < col+1 ; i=i+1) begin : col_num
      mac_tile #(.bw(bw), .psum_bw(psum_bw)) mac_tile_instance (
        .clk(clk),
        .reset(reset),
        .in_w( temp[bw*i-1:bw*(i-1)]),
        .out_e(temp[bw*(i+1)-1:bw*i]),
        .inst_w(inst_bus[2*i-1 : 2*(i-1)]),
        .inst_e(inst_bus[2*(i+1)-1 : 2*i]),
        .in_n(in_n[psum_bw*i-1 : psum_bw*(i-1)]),
        .is_os(is_os),
        .act_2b_mode(act_2b_mode),
        .out_s(out_s[psum_bw*i-1 : psum_bw*(i-1)]));

      assign valid[i-1] = inst_bus[2*(i+1)-1];
    end
  endgenerate
  
endmodule
