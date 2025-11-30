// Created by prof. Mingu Kang @VVIP Lab in UCSD ECE department
// Please do not spread this code without permission 
module mac_tile (clk, out_s, in_w, out_e, in_n, inst_w, inst_e, reset);
parameter bw = 4;
parameter psum_bw = 16;

output [psum_bw-1:0] out_s;
input  [bw-1:0] in_w; // inst[1]:execute, inst[0]: kernel loading
output [bw-1:0] out_e; 
input  [1:0] inst_w;
output [1:0] inst_e;
input  [psum_bw-1:0] in_n;
input  clk;
input  reset;

// states
reg load_ready_q, load_ready_nxt;
reg [1:0] inst_q, inst_nxt;
reg [bw-1:0] a_q, a_nxt;
reg [bw-1:0] b_q, b_nxt;
reg [psum_bw-1:0] c_q, c_nxt;


// output logic
assign out_e = a_q;
assign inst_e = inst_q;
mac #(.bw(bw), .psum_bw(psum_bw)) mac_instance(
	.out(out_s),
	.a(a_q),
	.b(b_q),
	.c(c_q)
);

// next state
always @(*) begin
	inst_nxt[0] = (~load_ready_q)? inst_w[0] : inst_q[0];
	inst_nxt[1] = inst_w[1];
	a_nxt       = (inst_w[0] | inst_w[1])   ? in_w : a_q;
	c_nxt = in_n;

	if (inst_w[0] & load_ready_q) begin
		b_nxt = in_w;
		load_ready_nxt = 0;
	end
	else begin
		b_nxt = b_q;
		load_ready_nxt = load_ready_q;
	end
end

// sequential logic
always @(posedge clk) begin
	if(reset)begin
		load_ready_q <= 1'b1;
		inst_q       <= 2'b00;
		a_q 		 <= {bw{1'b0}};
		b_q 		 <= {bw{1'b0}};
		c_q 		 <= {psum_bw{1'b0}};
	end
	else begin
		load_ready_q <= load_ready_nxt;
		inst_q       <= inst_nxt;
		a_q 		 <= a_nxt;
		b_q 		 <= b_nxt;
		c_q 		 <= c_nxt;
	end
end

endmodule
