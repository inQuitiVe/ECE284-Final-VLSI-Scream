module mac_array (clk, reset, out_s, in_w, in_n, inst_w, valid, is_os, act_4b_mode);

  parameter bw = 4;
  parameter psum_bw = 16;
  parameter col = 8;
  parameter row = 8;

  input  clk, reset;
  output [psum_bw*col-1:0] out_s;
  input  [row*bw-1:0] in_w;
  input  [2:0] inst_w; // WS: {reserved, execute, kernel loading} / OS: {flush psum, execute, psum loading}
  input  [psum_bw*col-1:0] in_n;
  output [col-1:0] valid;
  input  is_os; // whether this is in OS mode
  input  act_4b_mode;

  wire [psum_bw*col*row-1:0] out_s_temp;
  wire [col*row-1:0] valid_temp;
  wire [psum_bw*col*(row+1)-1:0] psum_temp;
  wire [(row+1)*col:0] in_n_zero_temp;

  assign psum_temp[psum_bw*col-1:0] = in_n;
  assign in_n_zero_temp[0] = !(&in_n[psum_bw*1-1:psum_bw*0]);
  assign in_n_zero_temp[1] = !(&in_n[psum_bw*2-1:psum_bw*1]);
  assign in_n_zero_temp[2] = !(&in_n[psum_bw*3-1:psum_bw*2]);
  assign in_n_zero_temp[3] = !(&in_n[psum_bw*4-1:psum_bw*3]);
  assign in_n_zero_temp[4] = !(&in_n[psum_bw*5-1:psum_bw*4]);
  assign in_n_zero_temp[5] = !(&in_n[psum_bw*6-1:psum_bw*5]);
  assign in_n_zero_temp[6] = !(&in_n[psum_bw*7-1:psum_bw*6]);
  assign in_n_zero_temp[7] = !(&in_n[psum_bw*8-1:psum_bw*7]);

  // assign out_s = out_s_temp[psum_bw*col*row-1:psum_bw*col*(row-1)];
  
  genvar j, k;
  generate
    for (j=0; j < col; j=j+1) begin : col_merge
      wire [psum_bw-1:0] col_psum [0:row-1]; // 2D array: psum from all rows in this column (after mask)
      
      for (k=0; k < row; k=k+1) begin : row_collect
        assign col_psum[k] = inst_sr[j][2] ? out_s_temp[psum_bw*(col*k+j+1)-1:psum_bw*(col*k+j)] : 0;
      end
      
      // OR all masked psums from all rows
      wire [psum_bw-1:0] or_result [0:row-1];
      genvar m;
      for (m=0; m < row; m=m+1) begin : or_tree
        if (m == 0) begin
          assign or_result[m] = col_psum[m];
        end else begin
          assign or_result[m] = or_result[m-1] | col_psum[m];
        end
      end
      
      // Output: use OR result in OS mode, otherwise use last row output
      assign out_s[psum_bw*(j+1)-1:psum_bw*j] = is_os ? or_result[row-1] : 
                                                 out_s_temp[psum_bw*(col*(row-1)+j+1)-1:psum_bw*(col*(row-1)+j)];
    end
  endgenerate
  
  // WS mode: valid from last row
  // OS mode: OR valid from all rows (if any row has valid, output is valid)
  genvar n, p;
  generate
    for (n=0; n < col; n=n+1) begin : valid_merge
      wire [row-1:0] col_valid; // valid from all rows for this column
      for (p=0; p < row; p=p+1) begin : valid_collect
        assign col_valid[p] = valid_temp[col*p+n];
      end
      assign valid[n] = is_os ? |col_valid : valid_temp[col*(row-1)+n];
    end
  endgenerate

  integer idx;
  reg [2:0] inst_sr [0:col-1];
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
          .out_s_zero(in_n_zero_temp[col*(i+2)-1:col*(i+1)]),
          .in_w(in_w[bw*(i+1)-1:bw*i]),
          .in_n(psum_temp[psum_bw*col*(i+1)-1:psum_bw*col*i]),
          .in_n_zero(in_n_zero_temp[col*(i+1)-1:col*i]),
          .valid(valid_temp[col*(i+1)-1:col*i]),
          .inst_w(inst_sr[i]),
          .is_os(is_os),
          .act_4b_mode(act_4b_mode)
        );
        assign psum_temp[psum_bw*col*(i+2)-1:psum_bw*col*(i+1)] = out_s_temp[psum_bw*col*(i+1)-1:psum_bw*col*i];
    end
  endgenerate


endmodule


