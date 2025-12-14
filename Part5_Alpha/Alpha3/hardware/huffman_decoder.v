module huffman_decoder (
    input wire clk,
    input wire rst_n,          // Active low reset
    input wire bit_in,         // Serial input bit
    input wire data_valid,     // High when bit_in is valid
    output reg [7:0] char_out, // Decoded symbol
    output reg char_valid      // High for 1 cycle when char_out is valid
);

    // Symbol Constants
    localparam [7:0] SYM_0 = 8'h00;
    localparam [7:0] SYM_1 = 8'h01;
    localparam [7:0] SYM_2 = 8'h02;
    localparam [7:0] SYM_3 = 8'h03;
    localparam [7:0] SYM_4 = 8'h04;
    localparam [7:0] SYM_5 = 8'h05;
    localparam [7:0] SYM_6 = 8'h06;
    localparam [7:0] SYM_7 = 8'h07;

    // State Encoding
    // We need deeper internal nodes to traverse this larger tree
    localparam [3:0] S_ROOT   = 4'd0;
    localparam [3:0] S_NODE_1 = 4'd1;
    localparam [3:0] S_NODE_2 = 4'd2;
    localparam [3:0] S_NODE_3 = 4'd3;
    localparam [3:0] S_NODE_4 = 4'd4;
    localparam [3:0] S_NODE_5 = 4'd5;
    localparam [3:0] S_NODE_6 = 4'd6;

    reg [3:0] state, next_state;
    reg [7:0] next_char;
    reg       next_valid;

    // Sequential Logic: State Updates
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state      <= S_ROOT;
            char_out   <= 8'h00;
            char_valid <= 1'b0;
        end else begin
            state      <= next_state;
            char_out   <= next_char;
            char_valid <= next_valid;
        end
    end

    // Combinational Logic: Tree Traversal
    always @(*) begin
        // Defaults
        next_state = state;
        next_char  = 8'h00;
        next_valid = 1'b0;

        if (data_valid) begin
            case (state)
                S_ROOT: begin
                    if (bit_in == 1'b0) begin
                        next_char  = SYM_0; // 0
                        next_valid = 1'b1;
                        next_state = S_ROOT;
                    end else begin
                        next_state = S_NODE_1;
                    end
                end

                S_NODE_1: begin
                    if (bit_in == 1'b0) begin
                        next_char  = SYM_1; // 10
                        next_valid = 1'b1;
                        next_state = S_ROOT;
                    end else begin
                        next_state = S_NODE_2;
                    end
                end

                S_NODE_2: begin
                    if (bit_in == 1'b0) begin
                        next_char  = SYM_2; // 110
                        next_valid = 1'b1;
                        next_state = S_ROOT;
                    end else begin
                        next_state = S_NODE_3;
                    end
                end

                S_NODE_3: begin
                    if (bit_in == 1'b0) begin
                        next_char  = SYM_3; // 1110
                        next_valid = 1'b1;
                        next_state = S_ROOT;
                    end else begin
                        next_state = S_NODE_4;
                    end
                end

                S_NODE_4: begin
                    if (bit_in == 1'b0) begin
                        next_char  = SYM_4; // 11110
                        next_valid = 1'b1;
                        next_state = S_ROOT;
                    end else begin
                        next_state = S_NODE_5;
                    end
                end

                S_NODE_5: begin
                    if (bit_in == 1'b0) begin
                        next_char  = SYM_5; // 111110
                        next_valid = 1'b1;
                        next_state = S_ROOT;
                    end else begin
                        next_state = S_NODE_6;
                    end
                end

                S_NODE_6: begin
                    if (bit_in == 1'b0) begin
                        next_char  = SYM_6; // 1111110
                        next_valid = 1'b1;
                        next_state = S_ROOT;
                    end else begin
                        next_char  = SYM_7; // 1111111
                        next_valid = 1'b1;
                        next_state = S_ROOT;
                    end
                end
                
                default: next_state = S_ROOT;
            endcase
        end
    end

endmodule