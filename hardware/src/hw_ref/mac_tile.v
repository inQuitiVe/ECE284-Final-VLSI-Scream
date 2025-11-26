module mac_tile (
    clk, 
    reset,
    in_w, 
    in_w_zero,
    in_n, 
    in_n_zero,
    inst_w, 
    inst_e, 
    act_4b_mode,
    is_os, // whether this is in OS mode
    out_s, 
    out_s_zero,
    out_e,
    out_e_zero
);

    parameter bw = 4;
    parameter psum_bw = 16;

    input                clk;
    input                reset;
    input  [bw-1:0]      in_w; 
    input                in_w_zero;
    input  [psum_bw-1:0] in_n;
    input                in_n_zero;
    input  [2:0]         inst_w; // WS: {reserved, execute, kernel loading} / OS: {flush psum, execute, psum loading}
    input                act_4b_mode;
    input                is_os; // whether this is in OS mode

    output [psum_bw-1:0] out_s;
    output reg           out_s_zero;
    output [bw-1:0]      out_e; 
    output reg           out_e_zero;
    output [2:0]         inst_e;

    // logic registers
    reg [2:0]            inst_q, inst_q_nxt;
    reg [bw-1:0]         a_q, a_q_nxt;      // activation
    reg [bw-1:0]         b_q, b_q_nxt;      // weight
    reg [psum_bw-1:0]    c_q, c_q_nxt;      // psum
    reg                  load_ready_q, load_ready_q_nxt; // WS: weight preload, OS: psum preload
    wire [psum_bw-1:0]   mac_out;
    wire                 is_preload;
    wire                 clk_x_gating;
    wire                 clk_w_gating;
    wire                 clk_psum_gating;
    wire                 in_n_zero_b;
    wire                 in_w_zero_b;  

    // Output assignments
    assign out_e  = a_q;
    assign inst_e = inst_q;
    // OS mode: when inst_q[2] (flush psum), output accumulated psum (c_q)
    assign out_s  = ~is_os ? mac_out : inst_q[2] ? c_q : in_n; 
    
    // WS mode: weight preload, OS mode: psum preload
    assign is_preload = inst_w[0] & load_ready_q;

    // Gating
    assign in_n_zero_b = !in_n_zero;
    assign in_w_zero_b = !in_w_zero;
    assign clk_x_gating = in_n_zero_b && clk;
    assign clk_w_gating = in_w_zero_b && clk;
    assign clk_psum_gating = (in_n_zero_b || in_w_zero_b) && clk;

    mac #(.bw(bw), .psum_bw(psum_bw)) mac_instance (
        .a(a_q), 
        .b(b_q),
        .c(c_q),
        .act_4b_mode(act_4b_mode),
        .out(mac_out)
    ); 

    // Combinational logic
    always @(*) begin
        inst_q_nxt[0]       = (load_ready_q == 0) ? inst_w[0] : inst_q[0];
        inst_q_nxt[1]       = inst_w[1];
        inst_q_nxt[2]       = inst_w[2]; // last data / flush psum

        load_ready_q_nxt    = is_preload ? 0 : load_ready_q;
        
        a_q_nxt             = (inst_w[0] || inst_w[1]) ? in_w : a_q;
        
        if (is_os) begin
            // When flush (inst_w[2]), don't update b to prevent receiving flushed psum
            b_q_nxt         = (inst_w[2] == 0 && (inst_w[0] || inst_w[1])) ? in_n[bw-1:0] : b_q;
            // preload psum when inst_w[0] && load_ready_q, save mac output when inst_w[1]
            // When flush (inst_w[2]), clear psum
            c_q_nxt         = inst_w[2] ? 0 : (is_preload ? in_n : (inst_w[1] ? mac_out : c_q));
        end else begin
            b_q_nxt         = is_preload ? in_w : b_q;
            c_q_nxt         = in_n;
        end
        
    end


    // Synchronous logic
    always @(posedge clk) begin
        if (reset) begin
            inst_q        <= 0;
            load_ready_q  <= 1;
            // a_q           <= 0;
            // b_q           <= 0;
            // c_q           <= 0;
            out_s_zero    <= 0;
            out_e_zero    <= 0;
        end else begin
            inst_q       <= inst_q_nxt;
            load_ready_q <= load_ready_q_nxt;
            // a_q          <= a_q_nxt;
            // b_q          <= b_q_nxt;
            // c_q          <= c_q_nxt;
            out_s_zero   <= in_n_zero;
            out_e_zero   <= in_w_zero;
        end
    end

    always @(posedge clk_x_gating || reset) begin
        if (reset) begin
            a_q <= 0;
        end
        else begin
            a_q <= a_q_nxt;
        end
    end

    always @(posedge clk_w_gating || reset) begin
        if (reset) begin
            b_q <= 0;
        end
        else begin
            b_q <= b_q_nxt;
        end
    end

    always @(posedge clk_psum_gating || reset) begin
        if (reset) begin
            c_q <= 0;
        end
        else begin
            c_q <= c_q_nxt;
        end
    end

endmodule


