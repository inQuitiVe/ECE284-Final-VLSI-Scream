// Created by prof. Mingu Kang @VVIP Lab in UCSD ECE department
// Please do not spread this code without permission 
module mac_array (clk, reset, out_s, in_w, in_n, inst_w, valid, is_os, act_2b_mode);

  parameter bw = 4;
  parameter psum_bw = 16;
  parameter col = 8;
  parameter row = 8;

  input  clk, reset;
  output [psum_bw*col-1:0] out_s;
  input  [row*bw-1:0] in_w; 
  input  [1:0] inst_w; // inst[1]:execute, inst[0]: kernel loading
  input  [psum_bw*col-1:0] in_n;
  output [col-1:0] valid;
  input  is_os;
  input  act_2b_mode;

  wire  [(row+1)*col*psum_bw-1:0] out_bus;
  assign out_bus[col*psum_bw-1:0] = in_n;

  wire [2*row-1:0] inst_bus;
  reg  [2*(row-1)-1:0] inst_reg ;
  assign inst_bus = {inst_reg, inst_w};


  // inst_w flows to row0 to row7
  always @ (posedge clk) begin
    if(reset)begin
      inst_reg <= {(row-1){2'b00}};
    end
    else begin
      inst_reg <= {inst_reg[2*(row-1)-3 : 0], inst_w};
    end
  end


  // valid collection
  wire [col*row-1:0] valid_bus;

  genvar i;
  generate
    for (i=1; i < row+1 ; i=i+1) begin : row_num
      mac_row #(.bw(bw), .psum_bw(psum_bw), .col(col)) mac_row_instance(
        .clk(clk), 
        .reset(reset), 
        .in_n(out_bus[i*col*psum_bw-1 : (i-1)*col*psum_bw]), 
        .out_s(out_bus[(i+1)*col*psum_bw-1 : i*col*psum_bw]),  
        .in_w(in_w[i*bw-1 : (i-1)*bw]), 
        .inst_w(inst_bus[i*2-1 : (i-1)*2]),
        .valid(valid_bus[col*i-1 : col*(i-1)]),
        .is_os(is_os),
        .act_2b_mode(act_2b_mode)
      );
    end
  endgenerate

  //output logic
  
  assign out_s = out_bus[(row+1)*col*psum_bw-1 : row*col*psum_bw];
  assign valid = valid_bus[col*row-1 : col*(row-1)];

endmodule
