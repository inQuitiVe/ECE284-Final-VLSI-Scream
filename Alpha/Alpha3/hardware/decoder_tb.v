`timescale 1ns / 1ps
`include "./hardware/huffman_decoder.v"

module huffman_tb;

    // --------------------------------------------------------
    // Signals
    // --------------------------------------------------------
    reg clk;
    reg rst_n;
    reg bit_in;
    reg data_valid;
    
    wire [7:0] char_out;
    wire char_valid;

    // --------------------------------------------------------
    // DUT Instantiation
    // --------------------------------------------------------
    huffman_decoder uut (
        .clk(clk),
        .rst_n(rst_n),
        .bit_in(bit_in),
        .data_valid(data_valid),
        .char_out(char_out),
        .char_valid(char_valid)
    );

    // --------------------------------------------------------
    // Clock Generation (100MHz equivalent)
    // --------------------------------------------------------
    initial clk = 0;
    always #5 clk = ~clk;

    // --------------------------------------------------------
    // Helper Tasks
    // --------------------------------------------------------
    
    // Task to drive a single bit into the decoder
    task drive_bit;
        input b;
        begin
            bit_in = b;
            data_valid = 1;
            @(posedge clk); // Hold for 1 clock cycle
            // data_valid remains high if we are streaming, 
            // but for safety in this task we can toggle it or keep it high.
            // Here we verify behavior by pulsing valid per bit.
            #1; // Hold time
            data_valid = 0;
            @(posedge clk); // Gap between bits (optional, verifies IDLE handling)
        end
    endtask

    // Task to send a full symbol based on the 0-7 codebook
    task send_symbol;
        input [2:0] sym;
        integer i;
        begin
            $display("[TB] Sending Symbol: %0d", sym);
            case (sym)
                3'd0: begin // Code: 0
                    $display("enocde as 0");
                    drive_bit(0);
                end
                3'd1: begin // Code: 10
                    $display("enocde as 10");
                    drive_bit(1); drive_bit(0);
                end
                3'd2: begin // Code: 110
                    $display("enocde as 110");
                    drive_bit(1); drive_bit(1); drive_bit(0);
                end
                3'd3: begin // Code: 1110
                    $display("enocde as 1110");
                    drive_bit(1); drive_bit(1); drive_bit(1); drive_bit(0);
                end
                3'd4: begin // Code: 11110
                    $display("enocde as 11110");
                    repeat(4) drive_bit(1);
                    drive_bit(0);
                end
                3'd5: begin // Code: 111110
                    $display("enocde as 111110");
                    repeat(5) drive_bit(1);
                    drive_bit(0);
                end
                3'd6: begin // Code: 1111110
                    $display("enocde as 1111110");
                    repeat(6) drive_bit(1);
                    drive_bit(0);
                end
                3'd7: begin // Code: 1111111
                    $display("enocde as 1111111");
                    repeat(7) drive_bit(1);
                end
            endcase
        end
    endtask

    // --------------------------------------------------------
    // Main Test Stimulus
    // --------------------------------------------------------
    initial begin
        // Initialize
        rst_n = 0;
        bit_in = 0;
        data_valid = 0;

        // Reset system
        #20 rst_n = 1;
        #20;

        $display("---------------------------------------");
        $display("Starting Huffman Decoder Simulation");
        $display("---------------------------------------");

        // Test Sequence: 0 -> 1 -> 7 -> 2 -> 5
        send_symbol(3'd0);  // Expect 0
        #20;
        
        send_symbol(3'd1);  // Expect 1
        #20;

        send_symbol(3'd7);  // Expect 7 (Boundary case: all 1s)
        #20;

        send_symbol(3'd2);  // Expect 2
        #20;
        
        send_symbol(3'd5);  // Expect 5
        #20;

        // Back-to-back testing (No delays between symbols)
        $display("[TB] Testing back-to-back streaming...");
        drive_bit(0); // Symbol 0
        drive_bit(1); drive_bit(0); // Symbol 1
        drive_bit(1); drive_bit(1); drive_bit(0); // Symbol 2

        #100;
        $display("---------------------------------------");
        $display("Simulation Complete");
        $display("---------------------------------------");
        $finish;
    end

    // --------------------------------------------------------
    // Output Monitor
    // --------------------------------------------------------
    always @(posedge clk) begin
        if (char_valid) begin
            $display("[DUT Output] Time: %0t | Decoded Symbol: %0d", $time, char_out);
        end
    end

endmodule