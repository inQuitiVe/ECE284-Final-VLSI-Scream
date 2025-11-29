module SFU #(
  parameter psum_bw  = 16,   // bit-width for partial sums
  parameter col      = 8
)(
    input clk,
    input reset,
    // sense signal from ofifo and output ctrl
    input                     ofifo_valid,
    input [psum_bw*col-1:0]   ofifo_data,
    // data in and ctrl signal for PSUM SRAM
    input  [psum_bw*col-1:0]  Q_pmem,  
    output ren_pmem,
    output wen_pmem,
    output [3:0]          r_A_pmem, //o_nij = 0~15
    output [3:0]          w_A_pmem,
    output [psum_bw*col-1:0]  D_pmem,
    // output
    input  [3:0]   kij
);

    

    // delayed signals
    reg [5:0]             nij_cnt; // 0~35
    


    reg                   ofifo_valid_D1;
    reg [psum_bw*col-1:0] ofifo_data_D1;



    reg  acc, acc_D1;
    reg [3:0] o_addr, o_addr_D1; // o_nij = 0~15
    // magic box
    integer row_a, col_a;
    integer k_row, k_col;
    integer o_row, o_col;
    integer tmp;
    always @(*) begin
        
        row_a = nij_cnt / 6;
        col_a = nij_cnt % 6;

        k_row = kij / 3;
        k_col = kij % 3;

        o_row = row_a - k_row;
        o_col = col_a - k_col;

        if (o_row >= 0 && o_row < 4 && o_col >= 0 && o_col < 4) begin
            acc = 1'b1;
            tmp     = o_row * 4 + o_col;
            o_addr  = tmp[3:0];   // 0..15
        end else begin
            acc = 1'b0;
            o_addr  = 4'd0;       // 隨便，反正不用
        end
    end

    // output

    

    // 


    // // Output logic
    assign ren_pmem = acc;
    assign wen_pmem = acc_D1;
    
    // The following expression is wrong !!!!
    // assign D_pmem = (kij==4'd0)? ofifo_data_D1 : (Q_pmem + ofifo_data_D1);
    assign D_pmem[15:0]    = (kij==4'd0)? ofifo_data_D1[15:0]    : (Q_pmem[15:0]    + ofifo_data_D1[15:0]   );   
    assign D_pmem[31:16]   = (kij==4'd0)? ofifo_data_D1[31:16]   : (Q_pmem[31:16]   + ofifo_data_D1[31:16]  );   
    assign D_pmem[47:32]   = (kij==4'd0)? ofifo_data_D1[47:32]   : (Q_pmem[47:32]   + ofifo_data_D1[47:32]  );   
    assign D_pmem[63:48]   = (kij==4'd0)? ofifo_data_D1[63:48]   : (Q_pmem[63:48]   + ofifo_data_D1[63:48]  );   
    assign D_pmem[79:64]   = (kij==4'd0)? ofifo_data_D1[79:64]   : (Q_pmem[79:64]   + ofifo_data_D1[79:64]  );   
    assign D_pmem[95:80]   = (kij==4'd0)? ofifo_data_D1[95:80]   : (Q_pmem[95:80]   + ofifo_data_D1[95:80]  );   
    assign D_pmem[111:96]  = (kij==4'd0)? ofifo_data_D1[111:96]  : (Q_pmem[111:96]  + ofifo_data_D1[111:96] );
    assign D_pmem[127:112] = (kij==4'd0)? ofifo_data_D1[127:112] : (Q_pmem[127:112] + ofifo_data_D1[127:112]);
    
    assign r_A_pmem = o_addr;
    assign w_A_pmem = o_addr_D1;
    

    //nij starts increment as o_valid = 1
    //r_A_pmem starts increment as o_valid = 1
    //w_A_pmem is has 1 cycle delay relative to r_A_pmem
    always @(posedge clk ) begin   
        if(reset | ~ofifo_valid) begin
            nij_cnt      <= 6'b00_0000;
            acc_D1       <= 1'b0;
            o_addr_D1    <= 4'b0000;
        end
        else begin
            nij_cnt      <= nij_cnt + 1'b1;
            acc_D1       <= acc;
            o_addr_D1    <= o_addr;
        end
    end 


    // delayed signals
    always @(posedge clk ) begin   
        if(reset) begin
            ofifo_valid_D1 <= 1'b0;
            ofifo_data_D1  <= {(psum_bw*col){1'b0}};
        end
        else begin
            ofifo_valid_D1 <= ofifo_valid;
            ofifo_data_D1  <= ofifo_data;
        end
    end 






    
    



    
endmodule