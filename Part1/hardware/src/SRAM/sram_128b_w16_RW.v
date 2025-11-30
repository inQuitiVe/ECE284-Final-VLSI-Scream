// Created by prof. Mingu Kang @VVIP Lab in UCSD ECE department
// Please do not spread this code without permission 
module sram_128b_w16_RW (CLK, D, Q, ren, wen, w_A, r_A);
// This SRAM can read and write at the same time!

  input  CLK;
  input  wen; //active high
  input  ren; //active high
  input  [127:0] D;
  input  [3:0] w_A;
  input  [3:0] r_A;
  output [127:0] Q;
  parameter num = 16;

   //   debug wire
   wire [15:0] out_och0_nij0 = memory[0][15:0];
   wire [15:0] out_och1_nij0 = memory[0][31:16];
   wire [15:0] out_och2_nij0 = memory[0][47:32];
   wire [15:0] out_och3_nij0 = memory[0][63:48];
   wire [15:0] out_och4_nij0 = memory[0][79:64];
   wire [15:0] out_och5_nij0 = memory[0][95:80];
   wire [15:0] out_och6_nij0 = memory[0][111:96];
   wire [15:0] out_och7_nij0 = memory[0][127:112];


  reg [127:0] memory [num-1:0];
  reg [3:0] add_q;
  assign Q = memory[add_q];

  always @ (posedge CLK) begin


   if(wen)begin // write
      memory[w_A] <= D; 
   end
   
   if(ren) begin // read 
      add_q <= r_A;
   end


  end

endmodule
