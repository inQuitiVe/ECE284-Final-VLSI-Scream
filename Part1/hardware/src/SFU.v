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
    output                    ren_pmem,
    output                    wen_pmem,
    output [3:0]              r_A_pmem, // o_nij = 0~15
    output [3:0]              w_A_pmem,
    output [psum_bw*col-1:0]  D_pmem,
    // kernel index
    input  [3:0]              kij,
    // readout phase
    input                     readout_start, // trigger for output stage
    output [psum_bw*col-1:0]  readout        // 16*8b
);

    // state
    reg [5:0]             nij_cnt;        // 0~35,
    reg readout_mode;
    reg [3:0]     readout_addr;        // 0~15



    // delayed signals
    reg                   ofifo_valid_D1;
    reg [psum_bw*col-1:0] ofifo_data_D1;
    reg                   acc, acc_D1;    // acc: Whether this psum is in output's region
    reg [3:0]             o_addr, o_addr_D1; // o_nij = 0~15


    // calculation o_addr from nij_cnt, kij
    // ######################## Magic Start #########################
    reg [3:0] row_a_hw, col_a_hw;
    reg [2:0] k_row_hw, k_col_hw;
    reg [3:0] o_row_hw, o_col_hw;

    // row_a = nij_cnt / 6; col_a = nij_cnt % 6;
    always @(*) begin
        case (nij_cnt)
            0,1,2,3,4,5:       row_a_hw = 4'd0;
            6,7,8,9,10,11:     row_a_hw = 4'd1;
            12,13,14,15,16,17: row_a_hw = 4'd2;
            18,19,20,21,22,23: row_a_hw = 4'd3;
            24,25,26,27,28,29: row_a_hw = 4'd4;
            30,31,32,33,34,35: row_a_hw = 4'd5;
            36,37,38,39,40,41: row_a_hw = 4'd6;
            42,43,44,45,46,47: row_a_hw = 4'd7;
            48,49,50,51,52,53: row_a_hw = 4'd8;
            54,55,56,57,58,59: row_a_hw = 4'd9;
            default          : row_a_hw = 4'd10;   //write down all in case debug need
        endcase
        col_a_hw = nij_cnt - row_a_hw * 6;
    end

    // k_row_hw = kij / 3; k_col_hw = kij % 3;
    always @(*) begin
        case (kij)
            0,1,2:    k_row_hw = 3'd0;
            3,4,5:    k_row_hw = 3'd1;
            6,7,8:    k_row_hw = 3'd2;
            9,10,11:  k_row_hw = 3'd3;
            12,13,14: k_row_hw = 3'd4;
            default:  k_row_hw = 3'd5;            //write down all in case debug need
        endcase
        k_col_hw = kij - k_row_hw * 3;
    end


    always @(*) begin
        o_row_hw = row_a_hw - k_row_hw;
        o_col_hw = col_a_hw - k_col_hw;
        if (o_row_hw >= 0 && o_row_hw < 4 && o_col_hw >= 0 && o_col_hw < 4) begin
            acc = 1'b1;
            o_addr  = o_row_hw * 4 + o_col_hw;  // 0..15
        end else begin
            acc = 1'b0;
            o_addr  = 4'd0;       // whatever
        end
    end

    // ######################## Magic End #########################





    // Output logic
    assign w_A_pmem = o_addr_D1;
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
    
    assign r_A_pmem = readout_mode? readout_addr : o_addr;
    assign ren_pmem = readout_mode? readout_mode : acc;
    assign readout = Q_pmem;
    

    //nij starts increment as o_valid = 1
    //r_A_pmem starts increment as o_valid = 1
    //w_A_pmem is has 1 cycle delay relative to r_A_pmem
    always @(posedge clk ) begin   
        if(reset) begin
            nij_cnt      <= 6'b00_0000;
        end
        else begin
            if(ofifo_valid)begin    // in case ofifo_valid was inserted bubble
                if(nij_cnt == 6'd36)begin
                    nij_cnt <= 6'd36;  // stuck nij here so that wen_pmem doesn't be pulled high
                end
                else begin
                    nij_cnt <= nij_cnt + 1'b1;
                end
            end
            else begin
                nij_cnt <= nij_cnt;
            end
        end
    end 

    always @(posedge clk ) begin
        if(reset) begin
            readout_mode   <= 1'b0;
            readout_addr <= 4'd0;
        end
        else begin
            if (!readout_mode && readout_start) begin //start readout
                readout_mode   <= 1'b1;
                readout_addr <= 1'b0;
            end
            else if(readout_mode && (readout_addr==4'd15)) begin //end readout
                readout_mode   <= 1'b0;
                readout_addr <= 4'd0;
            end
            else if(readout_mode) begin //keep readout
                readout_mode   <= 1'b1;
                readout_addr <= readout_addr + 1'b1;
            end
            else begin
                readout_mode   <= readout_mode;
                readout_addr <= readout_addr;
            end


        end
    end



    // delayed signals
    always @(posedge clk ) begin   
        if(reset) begin
            ofifo_valid_D1 <= 1'b0;
            ofifo_data_D1  <= {(psum_bw*col){1'b0}};
            acc_D1       <= 1'b0;
            o_addr_D1    <= 4'b0000;
        end
        else begin
            ofifo_valid_D1 <= ofifo_valid;
            ofifo_data_D1  <= ofifo_data;
            acc_D1       <= acc;
            o_addr_D1    <= o_addr;
        end
    end 






    
    



    
endmodule