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
    localparam S_Init    = 3'd0;
    localparam S_Acc     = 3'd1;
    localparam S_ReLU    = 3'd2;
    localparam S_Idle    = 3'd3;
    localparam S_Readout = 3'd4;

    // state
    reg flush_cycle, flush_cycle_nxt;
    reg [2:0]  state, state_nxt;


    reg [5:0]  nij, nij_nxt;       // 0~35,
    reg [3:0]  o_nij, o_nij_nxt;    // 0~15



    // combinational logic
    wire        cal_acc;  //Whether this psum is in output's region
    wire [3:0]  cal_o_nij;
    // calculation o_addr from nij, kij         
    onij_calculator onij_calculator_instance(
        .nij(nij), .kij(kij), .acc(cal_acc), .o_addr(cal_o_nij)
    );


    // delayed signals
    reg [psum_bw*col-1:0] ofifo_data_D1, ofifo_data_D2;
    reg [3:0]             r_A_pmem_D1;
    reg [3:0]             ren_pmem_D1;

    
    assign readout = Q_pmem;
    assign r_A_pmem = (state==S_Acc && cal_acc) ? cal_o_nij :
                      (state==S_ReLU || state==S_Readout) ? o_nij : 4'd0;
    assign ren_pmem = (state==S_Acc && cal_acc || state==S_ReLU || state==S_Readout) && (~flush_cycle);
    assign w_A_pmem = (state==S_Acc || state==S_ReLU) ? r_A_pmem_D1 : 4'd0;
    assign wen_pmem = (state==S_Acc || state==S_ReLU) ? ren_pmem_D1 : 1'b0;
    

    wire [psum_bw-1 : 0]  ReLU_out [col-1 : 0];
    genvar i;
    generate
        for(i=0; i<col; i=i+1)begin: col_num
            ReLU#(.psum_bw(psum_bw)) ReLU_instance(
                .in(Q_pmem[(i+1)*psum_bw-1 : i*psum_bw]),
                .out(ReLU_out[i])
            );
        end
    endgenerate

    generate
        for(i=0; i<col; i=i+1)begin
            assign D_pmem[(i+1)*psum_bw-1 : i*psum_bw] = (state==S_ReLU) ? ReLU_out[i] :
                                                         (state==S_Acc && kij==4'd0)  ?  ofifo_data_D2[(i+1)*psum_bw-1 : i*psum_bw] : $signed(Q_pmem[(i+1)*psum_bw-1 : i*psum_bw])+$signed(ofifo_data_D2[(i+1)*psum_bw-1 : i*psum_bw]);
        end
    endgenerate

    always @(*) begin
        case (state)
            S_Init: begin
                state_nxt = ofifo_valid ? S_Acc : S_Init;
                nij_nxt   = 6'd0;
                o_nij_nxt = 4'd0;
                flush_cycle_nxt = 1'b0;
            end
            S_Acc: begin
                if(flush_cycle)begin
                    flush_cycle_nxt = 1'b0;
                    nij_nxt = 6'd0;
                    state_nxt = (kij==4'd8)? S_ReLU : S_Idle;
                end
                else begin
                    if(nij == 6'd35)begin
                        nij_nxt = 6'd0;
                        flush_cycle_nxt = 1'b1;
                        state_nxt = S_Acc;
                    end
                    else begin
                        nij_nxt = nij + 1'b1;
                        flush_cycle_nxt = 1'b0;
                        state_nxt = S_Acc;
                    end
                end    
                o_nij_nxt = 4'd0;             
            end
            S_ReLU: begin
                if(flush_cycle)begin
                    flush_cycle_nxt = 1'b0;
                    o_nij_nxt = 4'd0;
                    state_nxt = S_Idle;
                end
                else begin
                    if(o_nij == 4'd15)begin
                        o_nij_nxt = 4'd0;
                        flush_cycle_nxt = 1'b1;
                        state_nxt = S_ReLU;
                    end
                    else begin
                        o_nij_nxt = o_nij + 1'b1;
                        flush_cycle_nxt = 1'b0;
                        state_nxt = S_ReLU;
                    end
                end           
                nij_nxt = 6'd0;      
            end
            S_Idle:begin
                state_nxt = readout_start? S_Readout : S_Idle;
                nij_nxt   = 6'd0;
                o_nij_nxt = 4'd0;
                flush_cycle_nxt = 1'b0;
            end
            S_Readout:begin
                if(flush_cycle)begin
                    flush_cycle_nxt = 1'b0;
                    o_nij_nxt = 4'd0;
                    state_nxt = S_Idle;
                end
                else begin
                    if(o_nij == 4'd15)begin
                        o_nij_nxt = 4'd0;
                        flush_cycle_nxt = 1'b1;
                        state_nxt = S_Readout;
                    end
                    else begin
                        o_nij_nxt = o_nij + 1'b1;
                        flush_cycle_nxt = 1'b0;
                        state_nxt = S_Readout;
                    end
                end           
                nij_nxt = 6'd0; 
            end
            default: begin
                state_nxt = S_Init;
                nij_nxt   = 6'd0;
                o_nij_nxt = 4'd0;
                flush_cycle_nxt = 1'b0;
            end
        endcase
    end



    // state logic
    always @(posedge clk ) begin
        if(reset) begin
            state <= S_Init;
            nij   <= 6'b0;
            o_nij <= 4'b0;
            flush_cycle <= 1'b0;
        end
        else begin
            state <= state_nxt;
            nij   <= nij_nxt;
            o_nij <= o_nij_nxt;
            flush_cycle <= flush_cycle_nxt;
        end
    end


    // delayed signals
    always @(posedge clk ) begin   
        if(reset) begin
            ofifo_data_D1  <= {(psum_bw*col){1'b0}};
            ofifo_data_D2  <= {(psum_bw*col){1'b0}};
            r_A_pmem_D1    <= 4'd0;
            ren_pmem_D1    <= 1'b0;
        end
        else begin
            ofifo_data_D1  <= ofifo_data;
            ofifo_data_D2  <= ofifo_data_D1;
            r_A_pmem_D1    <= r_A_pmem;
            ren_pmem_D1    <= ren_pmem;
        end
    end 






    



    
endmodule