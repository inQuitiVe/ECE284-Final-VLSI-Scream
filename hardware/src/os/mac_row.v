module mac_row (clk, out_s, in_w, in_n, valid, inst_w, reset, is_os, act_4b_mode);

  parameter bw = 4;
  parameter psum_bw = 16;
  parameter col = 8;

  input  clk, reset;
  output [psum_bw*col-1:0] out_s;
  output [col-1:0] valid;
  input  [bw-1:0] in_w;
  input  [2:0] inst_w; // WS: {reserved, execute, kernel loading} / OS: {flush psum, execute, psum loading}
  input  [psum_bw*col-1:0] in_n;
  input  is_os; // whether this is in OS mode
  input  act_4b_mode;

  wire  [(col+1)*bw-1:0] temp;
  wire  [(col+1)*3-1:0] inst_temp;

  assign temp[bw-1:0]   = in_w;
  assign inst_temp[2:0] = inst_w;

  genvar i;
  generate
    for (i=0; i < col ; i=i+1) begin : col_num
        mac_tile #(.bw(bw), .psum_bw(psum_bw)) mac_tile_instance (
          .clk(clk),
          .reset(reset),
          .in_w( temp[bw*(i+1)-1:bw*i]),
          .out_e(temp[bw*(i+2)-1:bw*(i+1)]),
          .inst_w(inst_temp[3*(i+1)-1:3*i]),
          .inst_e(inst_temp[3*(i+2)-1:3*(i+1)]),
          .in_n(in_n[psum_bw*(i+1)-1:psum_bw*i]),
          .out_s(out_s[psum_bw*(i+1)-1:psum_bw*i]),
          .is_os(is_os),
          .act_4b_mode(act_4b_mode)
        );
        // WS: valid = execute bit, OS: valid = flush psum bit (both from inst_e)
        assign valid[i] = is_os ? inst_temp[3*(i+1)+2] : inst_temp[3*(i+1)+1];
    end
  endgenerate

endmodule