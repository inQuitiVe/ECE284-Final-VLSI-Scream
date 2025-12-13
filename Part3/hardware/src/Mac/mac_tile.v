module mac_tile (
  clk,
  reset,
  in_w,
  in_n,
  inst_w,
  inst_e,
  act_2b_mode,
  is_os,  // whether this is in OS mode
  out_s,
  out_e
);

  parameter bw      = 4;
  parameter psum_bw = 16;

  input                clk;
  input                reset;
  input  [bw-1:0]      in_w;
  input  [psum_bw-1:0] in_n;
  input  [2:0]         inst_w;  // WS: {reserved, execute, kernel loading} / OS: {flush psum, execute, psum loading}
  input                act_2b_mode;
  input                is_os;  // whether this is in OS mode

  output [psum_bw-1:0] out_s;
  output [bw-1:0]      out_e;
  output [2:0]         inst_e;

  // logic registers
  reg [2:0]          inst_q, inst_q_nxt;
  reg [bw-1:0]       a_q, a_q_nxt;  // activation
  reg [bw-1:0]       b0_q, b0_q_nxt, b1_q, b1_q_nxt;  // weight
  reg [psum_bw-1:0]  c_q, c_q_nxt;  // psum
  // load_ready_q: init to 2'b11 for WS in 2 bit mode, load to b0 when 3, load b1 when 1
  reg  [1:0]         load_ready_q, load_ready_q_nxt;  // WS: weight preload, OS: psum preload
  wire [psum_bw-1:0] mac_out;
  wire               is_preload;

  // Output assignments
  assign out_e  = a_q;
  assign inst_e = inst_q;
  // OS mode: when inst_q[2] (flush psum), output accumulated psum (c_q)
  assign out_s  = ~is_os ? mac_out : inst_w[2] ? c_q : act_2b_mode ? {8'b0, b1_q, b0_q} : {12'b0, b0_q};

  // WS mode: weight preload, OS mode: psum preload
  assign is_preload = inst_w[0] & |load_ready_q;

  mac #(.bw(bw), .psum_bw(psum_bw)) mac_instance (
    .a           (a_q           ),
    .b0          (b0_q          ),
    .b1          (b1_q          ),
    .c           (c_q           ),
    .act_2b_mode (act_2b_mode   ),
    .out         (mac_out       )
  ); 

  // Combinational logic
  always @(*) begin
    inst_q_nxt[0] = (load_ready_q == 0) ? inst_w[0] : inst_q[0];
    inst_q_nxt[1] = inst_w[1];
    inst_q_nxt[2] = inst_w[2];  // last data / flush psum

    load_ready_q_nxt = is_preload ? load_ready_q-1 : load_ready_q;

    a_q_nxt = (inst_w[0] || inst_w[1]) ? in_w : 0;

    if (is_os) begin
      // When flush (inst_w[2]), don't update b to prevent receiving flushed psum
      b0_q_nxt = (inst_w[2] == 0 && (inst_w[0] || inst_w[1])) ? in_n[bw-1:0] : b0_q;
      if (act_2b_mode) begin
        b1_q_nxt = (inst_w[2] == 0 && (inst_w[0] || inst_w[1])) ? in_n[bw-1:0] : b1_q;
      end
      else begin
        b1_q_nxt = (inst_w[2] == 0 && (inst_w[0] || inst_w[1])) ? in_n[2*bw-1:0] : b1_q;
      end
      // preload psum when inst_w[0] && load_ready_q, save mac output when inst_w[1]
      // When flush (inst_w[2]), clear psum
      c_q_nxt = inst_w[2] ? 0 : (is_preload ? in_n : (inst_q[1] ? mac_out : c_q));
    end
    else begin
      if (~act_2b_mode) begin
        b0_q_nxt = is_preload ? in_w : b0_q;
        b1_q_nxt = is_preload ? in_w : b1_q;
      end
      else begin
        b0_q_nxt = (inst_w[0] && load_ready_q == 2'b01) ? in_w : b0_q;
        b1_q_nxt = (inst_w[0] && load_ready_q == 2'b10) ? in_w : b1_q;
      end
      c_q_nxt = in_n;
    end
  end


  // Synchronous logic
  always @ (posedge clk) begin
    if (reset) begin
      inst_q       <= 0;
      load_ready_q <= (~is_os & act_2b_mode) ? 2'b10 : 2'b01;
      a_q          <= 0;
      b0_q         <= 0;
      b1_q         <= 0;
      c_q          <= 0;
    end
    else begin
      inst_q       <= inst_q_nxt;
      load_ready_q <= load_ready_q_nxt;
      a_q          <= a_q_nxt;
      b0_q         <= b0_q_nxt;
      b1_q         <= b1_q_nxt;
      c_q          <= c_q_nxt;
    end
  end

endmodule