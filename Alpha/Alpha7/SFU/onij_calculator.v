module onij_calculator (
    input  [5:0]  nij, // 0~35
    input  [3:0]  kij, // 0~8
    output reg       acc, //whether this psum[kij, nij, :] falls in output region
    output reg [3:0]  o_addr //if yes, the o_nij. 0~15
);


    reg [3:0] row_a_hw, col_a_hw;
    reg [2:0] k_row_hw, k_col_hw;
    reg [3:0] o_row_hw, o_col_hw;

    // row_a = nij / 6; col_a = nij % 6;
    always @(*) begin
        case (nij)
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
        col_a_hw = nij - row_a_hw * 6;
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

    
endmodule