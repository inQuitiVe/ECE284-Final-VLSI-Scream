module ActivationFunction #(
    parameter psum_bw  = 16   // bit-width for partial sums
) (
    input  [psum_bw-1: 0] in,
    input  [1:0]          act_func_mode,  // 00: ReLU, 01: ELU, 10: LeakyReLU, 11: GELU
    output [psum_bw-1: 0] out
);

    // Activation function mode definitions
    localparam ACT_RELU     = 2'b00;
    localparam ACT_ELU      = 2'b01;
    localparam ACT_LEAKYRELU = 2'b10;
    localparam ACT_GELU     = 2'b11;

    // ReLU: max(0, x) = x if x >= 0, else 0
    wire [psum_bw-1:0] relu_out;
    assign relu_out = in[psum_bw-1] ? {psum_bw{1'b0}} : in;

    // Signed input for arithmetic operations
    wire signed [psum_bw-1:0] in_signed;
    assign in_signed = in;

    // LeakyReLU: x if x > 0, else 0.01 * x
    // For hardware implementation, we use fixed-point: 0.01 ≈ 1/64 = 2^-6
    // LeakyReLU(x) = x if x >= 0, else (x >> 6) for negative values
    wire [psum_bw-1:0] leakyrelu_out;
    wire signed [psum_bw-1:0] leakyrelu_signed;
    assign leakyrelu_signed = in_signed[psum_bw-1] ? (in_signed >>> 6) : in_signed;
    assign leakyrelu_out = leakyrelu_signed;

    // ELU: x if x > 0, else α * (e^x - 1) where α = 1.0
    // Hardware approximation using piecewise linear segments
    // For negative values: ELU(x) ≈ -1 for x < -4, linear interpolation for -4 <= x < 0
    wire [psum_bw-1:0] elu_out;
    wire signed [psum_bw-1:0] elu_signed;
    wire signed [psum_bw-1:0] elu_linear;
    assign elu_linear = (in_signed >>> 2) - 1;  // Linear approximation: ELU(x) ≈ x/4 - 1 for -4 <= x < 0
    assign elu_signed = (in_signed >= 0) ? in_signed :
                        (in_signed < -4) ? -1 : elu_linear;  // Saturate at -1 for very negative values
    assign elu_out = elu_signed;

    // GELU: 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
    // Hardware approximation using piecewise linear segments
    // Simplified: GELU(x) ≈ 0.5 * x * (1 + sign(x)) for small |x|
    wire [psum_bw-1:0] gelu_out;
    wire signed [psum_bw-1:0] gelu_signed;
    wire signed [psum_bw-1:0] gelu_temp1, gelu_temp2;
    // GELU approximation: for x > 0: GELU(x) ≈ x * 0.5
    //                    for x < 0: GELU(x) ≈ 0 (saturate)
    // Better approximation would use polynomial or LUT
    assign gelu_temp1 = in_signed >>> 1;  // x * 0.5
    assign gelu_temp2 = (in_signed < 0) ? 0 : gelu_temp1;
    assign gelu_signed = gelu_temp2;
    assign gelu_out = gelu_signed;

    // Select output based on activation function mode
    assign out = (act_func_mode == ACT_RELU)     ? relu_out :
                 (act_func_mode == ACT_ELU)      ? elu_out :
                 (act_func_mode == ACT_LEAKYRELU) ? leakyrelu_out :
                 (act_func_mode == ACT_GELU)     ? gelu_out :
                 relu_out;  // Default to ReLU

endmodule

