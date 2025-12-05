// Created by prof. Mingu Kang @VVIP Lab in UCSD ECE department
// Please do not spread this code without permission 
module max4 (clk, reset, in, take_MPL, out);

  parameter bw = 4;

  input  clk;
  input  reset;

  input  [bw-1:0] in;
  input   take_MPL;
  output [bw-1:0] out;

  reg [bw-1:0] q0;
  reg [bw-1:0] q1;
  reg [bw-1:0] q2;
  reg [bw-1:0] q3;

    wire [bw-1:0] max2_0;
    wire [bw-1:0] max2_1;
    wire [bw-1:0] max4;


     assign max2_0 = in>q0 ? in : q0;
     assign max2_1 = q1>q2 ? q1 : q2;
     assign max4   = max2_0>max2_1 ? max2_0 : max2_1;
     assign out    = take_MPL? max4 : {bw{1'b0}};


 always @ (posedge clk) begin
   if (reset) begin
        q0 <= {bw{1'b0}};
        q1 <= {bw{1'b0}};
        q2 <= {bw{1'b0}};
        q3 <= {bw{1'b0}};
   end
   else begin
        q0 <= in;
        q1 <= q0;
        q2 <= q1;
        q3 <= q2;
   end
 end


endmodule
