// Created by prof. Mingu Kang @VVIP Lab in UCSD ECE department
// Please do not spread this code without permission 
module mac_array (clk, reset, out_s, in_w, in_n, inst_w, valid, is_os, act_2b_mode);

  parameter bw      = 4;
  parameter psum_bw = 16;
  parameter col     = 8;
  parameter row     = 8;
  parameter inst_bw = 3;

  input  clk, reset;
  output [psum_bw*col-1:0] out_s;
  input  [row*bw-1:0]      in_w;
  input  [inst_bw-1:0]             inst_w;  // inst[1]:execute, inst[0]: kernel loading
  input  [psum_bw*col-1:0] in_n;
  output [col-1:0]         valid;
  input                    is_os;
  input                    act_2b_mode;

  wire [(row+1)*col*psum_bw-1:0] out_bus;
  assign out_bus[col*psum_bw-1:0] = in_n;

  wire [inst_bw*(row+col+1)-1:0]      inst_bus;
  reg  [inst_bw*(row+col)-1:0]  inst_reg;
  assign inst_bus = {inst_reg, inst_w};


  // inst_w flows to row0 to row7
  always @ (posedge clk) begin
    if (reset) begin
      inst_reg <= {(inst_bw*(row+col)){1'b0}};
    end
    else begin
      inst_reg <= inst_bus[inst_bw*(row+col)-1:0];
    end
  end


  // valid collection
  wire [col*row-1:0] valid_bus;

  genvar i;
  generate
    for (i=1; i<row+1; i=i+1) begin : row_num
      mac_row #(.bw(bw), .psum_bw(psum_bw), .col(col)) mac_row_instance (
        .clk        (clk                                                      ),
        .reset      (reset                                                    ),
        .in_n       (out_bus[i*col*psum_bw-1:(i-1)*col*psum_bw]              ),
        .out_s      (out_bus[(i+1)*col*psum_bw-1:i*col*psum_bw]              ),
        .in_w       (in_w[i*bw-1:(i-1)*bw]                                    ),
        .inst_w     (inst_bus[i*inst_bw-1:(i-1)*inst_bw]                         ),
        .valid      (valid_bus[col*i-1:col*(i-1)]                             ),
        .is_os      (is_os                                                    ),
        .act_2b_mode(act_2b_mode                                              )
      );
    end
  endgenerate

  // output logic
  // assign out_s = out_bus[(row+1)*col*psum_bw-1:row*col*psum_bw];
  // assign valid  = valid_bus[col*row-1:col*(row-1)];
  genvar j, k;
  generate
    for (j=0; j < col; j=j+1) begin : col_merge
      // Flatten 2D array: col_psum from all rows in this column (after mask)
      wire [psum_bw*row-1:0] col_psum_flat;
      wire [inst_bw*row-1:0] row_k_inst;
      
      for (k=0; k < row; k=k+1) begin : row_collect
        
        assign row_k_inst[k*inst_bw+:inst_bw] = inst_bus[(j+k+1)*inst_bw+:inst_bw];
        
        wire [psum_bw-1:0] row_k_col_j_psum;
        wire [psum_bw*col-1:0] partial_out_bus = out_bus[(k+1)*col*psum_bw+:psum_bw*col];
        assign row_k_col_j_psum = out_bus[(k+1)*col*psum_bw+psum_bw*j+:psum_bw];

        
        // Mask: if inst[2] (flush psum) is set, use the psum, otherwise 0
        assign col_psum_flat[psum_bw*(k+1)-1:psum_bw*k] = row_k_inst[k*inst_bw+2] ? row_k_col_j_psum : {psum_bw{1'b0}};
      end
      
      // OR all masked psums from all rows (flatten tree)
      wire [psum_bw*row-1:0] or_result_flat;
      genvar m;
      for (m=0; m < row; m=m+1) begin : or_tree
        if (m == 0) begin
          assign or_result_flat[psum_bw-1:0] = col_psum_flat[psum_bw-1:0];
        end else begin
          assign or_result_flat[psum_bw*(m+1)-1:psum_bw*m] = 
                 or_result_flat[psum_bw*m-1:psum_bw*(m-1)] | col_psum_flat[psum_bw*(m+1)-1:psum_bw*m];
        end
      end
      
      // Output: use OR result in OS mode, otherwise use last row output
      // Last row (row) output for column j is at out_bus[row*col*psum_bw+psum_bw*(j+1)-1:row*col*psum_bw+psum_bw*j]
      wire [psum_bw-1:0] last_row_col_j_psum;
      assign last_row_col_j_psum = out_bus[row*col*psum_bw+psum_bw*(j+1)-1:row*col*psum_bw+psum_bw*j];

      wire [psum_bw-1:0] partial_out_s = is_os ? or_result_flat[psum_bw*row-1:psum_bw*(row-1)] : last_row_col_j_psum;      
      assign out_s[psum_bw*(j+1)-1:psum_bw*j] = partial_out_s;
    end
  endgenerate
  
  // WS mode: valid from last row
  // OS mode: OR valid from all rows (if any row has valid, output is valid)
  genvar n, p;
  generate
    for (n=0; n < col; n=n+1) begin : valid_merge
      // Flatten: collect valid from all rows for this column
      wire [row-1:0] col_valid_flat;
      for (p=0; p < row; p=p+1) begin : valid_collect
        // valid_bus format: [col*row-1:0] where valid_bus[col*p+col-1:col*p] is row p's valid
        // Column n in row p is at valid_bus[col*p+n]
        assign col_valid_flat[p] = valid_bus[col*p+n];
      end
      // Last row (row-1) valid for column n
      wire last_row_col_n_valid;
      assign last_row_col_n_valid = valid_bus[col*(row-1)+n];
      
      assign valid[n] = is_os ? |col_valid_flat : last_row_col_n_valid;
    end
  endgenerate

endmodule
