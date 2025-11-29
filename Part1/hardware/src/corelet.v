module corelet #(
  parameter bw       = 4,    // bit-width for weights/activations
  parameter psum_bw  = 16,   // bit-width for partial sums
  parameter col      = 8,
  parameter row      = 8
)(
    input           clk,
    input           reset,

    // high level instructions from TB
    input   [1:0]   inst_w,
    input   [row*bw-1:0] vector_data_in
    
    // SFU and OFIFO is somewhat more complex
);


// L0fifo ctrl signal
reg [1:0] inst_w_D1, inst_w_D2;
always @(posedge clk ) begin
    if(reset)begin
        inst_w_D1 <= 2'b00;
        inst_w_D2 <= 2'b00;
    end
    else begin
        inst_w_D1 <= inst_w;   // start reading out data after writing
        inst_w_D2 <= inst_w_D1;
    end
end


// L0 fifo output
wire [row*bw-1:0]     mac_array_data_in;



l0 #(.bw(bw), .row(row)) L0fifo_instance(
    .clk(clk),
    .reset(reset), 
    .in(vector_data_in), 
    .out(mac_array_data_in), 
    .rd(|inst_w_D1), // start reading out data after writing
    .wr(|inst_w), // either kernel_load or execution, capture data
    .o_full(), 
    .o_ready()
);

mac_array #(.bw(bw), .psum_bw(psum_bw), .col(col), .row(row))mac_array_instance(
    .clk(clk), 
    .reset(reset), 
    .out_s(), 
    .in_w(mac_array_data_in), 
    .in_n(), 
    .inst_w(inst_w_D2), 
    .valid()
);




endmodule