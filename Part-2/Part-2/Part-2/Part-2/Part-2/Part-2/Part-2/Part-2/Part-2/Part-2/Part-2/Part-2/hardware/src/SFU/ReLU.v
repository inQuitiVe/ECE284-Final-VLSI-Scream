module ReLU #(
    parameter psum_bw  = 16   // bit-width for partial sums
) (
    input  [psum_bw-1: 0] in,
    output [psum_bw-1: 0] out
);

    assign out = in[psum_bw-1] ? {psum_bw{1'b0}} : in;
    // assign out = in;
endmodule