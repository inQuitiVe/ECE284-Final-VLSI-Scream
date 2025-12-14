module MPL2D #(
    parameter psum_bw  = 16,
    parameter col      = 8
)(
    input clk,
    input reset,
    input enable,
    input  [3:0]  order, // 0~15
    input  [psum_bw*col-1:0] in,
    output [3:0]  o_nij, // 0~15
    output [1:0]  mpl_onij, // 0~15
    output       MPL_valid,
    output reg [psum_bw*col-1:0] out
    
);

    assign o_nij    = {order[3], order[1], order[2], order[0]};
    // This simpling wiring pattern is found by K-map, which is very beautiful


    reg  [3:0]  order_D1;

    // order 0~3 is the mpl_onij=2'b00
    // order 4~7 is the mpl_onij=2'b01
    // order 8~11 is the mpl_onij=2'b10
    // order 12~15 is the mpl_onij=2'b11
    wire MPL_subcycle_start = enable && ~|order_D1[1:0];
    assign mpl_onij = order_D1[3:2];
    assign MPL_valid = enable && &order_D1[1:0];


    reg [psum_bw*col-1:0] out_q;

    integer i;
    always @(*) begin
        for(i=0; i<col; i=i+1)begin
            if(MPL_subcycle_start) begin
                out[i*psum_bw +: psum_bw]  = in[i*psum_bw +: psum_bw]; 
            end
            else begin
                if($signed(in[i*psum_bw +: psum_bw]) >= $signed(out_q[i*psum_bw +: psum_bw])) begin
                    out[i*psum_bw +: psum_bw]  = in[i*psum_bw +: psum_bw];
                end
                else begin
                    out[i*psum_bw +: psum_bw]  = out_q[i*psum_bw +: psum_bw];
                end
            end


            
        end
    end

    always @(posedge clk) begin
        if(reset || ~enable)begin
            out_q <=   {8{{1'b1}, {(psum_bw-1){1'b0}}}};
        end
        else begin
            out_q <= out;
        end
    end
    

    // delayed signal
    always @(posedge clk) begin
        if(reset)begin
            order_D1 <= 4'd0;
        end
        else begin
            order_D1 <= order;
        end
    end

    
    


    // debug wire
    wire [psum_bw-1:0] out_col0 = out[15:0];
    wire [psum_bw-1:0] out_col1 = out[31:16];
    wire [psum_bw-1:0] out_col2 = out[47:32];
    wire [psum_bw-1:0] out_col3 = out[63:48];
    wire [psum_bw-1:0] out_col4 = out[79:64];
    wire [psum_bw-1:0] out_col5 = out[95:80];
    wire [psum_bw-1:0] out_col6 = out[111:96];
    wire [psum_bw-1:0] out_col7 = out[127:112];

    wire [psum_bw-1:0] in_col0 = in[15:0];
    wire [psum_bw-1:0] in_col1 = in[31:16];
    wire [psum_bw-1:0] in_col2 = in[47:32];
    wire [psum_bw-1:0] in_col3 = in[63:48];
    wire [psum_bw-1:0] in_col4 = in[79:64];
    wire [psum_bw-1:0] in_col5 = in[95:80];
    wire [psum_bw-1:0] in_col6 = in[111:96];
    wire [psum_bw-1:0] in_col7 = in[127:112];



    

    
endmodule