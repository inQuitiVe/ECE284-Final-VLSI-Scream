module mac (out, a, b, c, act_4b_mode);

    parameter bw = 4;
    parameter psum_bw = 16;
    
    output signed [psum_bw-1:0] out;
    input signed  [bw-1:0]      a;  // activation
    input signed  [bw-1:0]      b0, b1;  // weight
    input signed  [psum_bw-1:0] c;
    input                       act_4b_mode;   // 0 = 2-bit act, 1 = 4-bit act

    wire          [2:0]         a_lo_pad, a_hi_pad;
    wire signed   [psum_bw-1:0] psum;
    wire signed   [6:0]         product_lo, product_hi;
    wire signed   [15:0]        product_lo_pad, product_hi_pad;
    wire signed   [15:0]        product;

    assign a_hi_pad   = {1'b0, a[3:2]}; // force to be unsigned number
    assign a_lo_pad   = {1'b0, a[1:0]}; // force to be unsigned number

    assign product_lo = $signed(a_lo_pad) * $signed(b0);
    assign product_hi = $signed(a_hi_pad) * $signed(b1);

    assign product_lo_pad = {{2{product_lo[6]}}, product_lo[5:0]};
    assign product_hi_pad = {product_hi[5:0]   ,            2'b0};

    assign product    = act_4b_mode ? $signed(product_lo_pad) + $signed(product_hi_pad) : $signed(product_lo) + $signed(product_hi);

    assign psum       = $signed(product) + $signed(c);
    assign out        = psum;

endmodule   
