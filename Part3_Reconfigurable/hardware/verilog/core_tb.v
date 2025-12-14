// Created by prof. Mingu Kang @VVIP Lab in UCSD ECE department
// Please do not spread this code without permission 
`timescale 1ns/1ps

// Mode selection:
//   - Define ACT_2BIT for 2-bit activation mode (act_2b_mode = 1)
//   - Undefine ACT_2BIT for 4-bit activation mode (act_2b_mode = 0, default)
//   - Define IS_OS for Output Stationary mode (is_os = 1)
//   - Undefine IS_OS for Weight Stationary mode (is_os = 0, default)


module core_tb;

  parameter bw      = 4;
  parameter psum_bw = 16;
  parameter len_kij = 9;
  parameter len_onij = 16;
  parameter col     = 8;
  parameter mac_col = 8;
  parameter row     = 8;
  parameter len_nij = 36;

  reg clk   = 0;
  reg reset = 1;

  // High level instrucion & data for mac array.
  reg [1:0]        inst_w;
  reg [bw*row-1:0] D_xmem;
  reg              is_os;
  reg              act_2b_mode;

  // X_MEM Ctrl (for activation.txt and weight.txt writing)
  reg        CEN_xmem = 1;
  reg        WEN_xmem = 1;
  reg [10:0] A_xmem   = 0;

  // Readout (for output.txt verification. tb has no access to PSUM mem)
  reg                   readout_start;
  wire [psum_bw*mac_col-1:0] readout;

  // Control Signal for SFU. I hope to get rid of this in the future versions.
  reg [3:0] kij_SFUctrl;

  // Misc
  reg [8*30:1]          stringvar;
  reg [8*50:1]          w_file_name;
  reg [8*50:1]          w1_file_name;  // file name for tile1 weight
  reg [psum_bw*col-1:0] answer;
  integer x_file, x_scan_file;   // file_handler
  integer x1_file, x1_scan_file; // file_handler for tile1 activation
  integer w_file, w_scan_file;   // file_handler
  integer w1_file, w1_scan_file; // file_handler for tile1 weight
  integer out_file, out_scan_file;  // file_handler
  integer captured_data;
  integer t, i, j, k, kij;
  integer error;
  reg [15:0] tile0_data, tile1_data;  // temporary storage for interleaving


  core #(.bw(bw), .psum_bw(psum_bw), .col(mac_col), .row(row)) core_instance (
    .clk          (clk                ),
    .reset        (reset              ),
    .inst_w       ({1'b0, inst_w}             ),  // high level instructions from TB
    .CEN_xmem     (CEN_xmem           ),  // x_mem ctrls from TB
    .WEN_xmem     (WEN_xmem           ),
    .A_xmem       (A_xmem             ),
    .D_xmem       (D_xmem             ),
    .is_os        (is_os              ),
    .act_2b_mode  (act_2b_mode        ),
    .kij          (kij_SFUctrl        ),  // SFU control
    .readout_start(readout_start      ),  // Output to TB
    .readout      (readout            )
  ); 


  initial begin
    inst_w       = 0;
    D_xmem       = 0;
    CEN_xmem     = 1;
    WEN_xmem     = 1;
    A_xmem       = 0;
    kij_SFUctrl  = 0;
    readout_start = 0;
    `ifdef IS_OS
        is_os        = 1;  // Output Stationary mode
        $display("## Running in OS mode (IS_OS defined)");
    `else
        is_os        = 0;  // Weight Stationary mode (default)
        $display("## Running in WS mode (IS_OS not defined)");
    `endif
    `ifdef ACT_2BIT
        act_2b_mode  = 1;  // 2-bit activation mode
        $display("## Running in 2-bit activation mode (ACT_2BIT defined)");
    `else
        act_2b_mode  = 0;  // 4-bit activation mode (default)
        $display("## Running in 4-bit activation mode (ACT_2BIT not defined)");
    `endif

    `ifdef ACT_2BIT
      $dumpfile("core_tb_2bit.vcd");
      $dumpvars(0, core_tb);
    `else
      $dumpfile("core_tb_vanilla.vcd");
      $dumpvars(0, core_tb);
    `endif

    `ifdef ACT_2BIT
      x_file = $fopen("../datafiles/ws2bit/activation_tile0.txt", "r");
      x1_file = $fopen("../datafiles/ws2bit/activation_tile1.txt", "r");
    `else
      x_file = $fopen("../datafiles/ws4bit/activation_tile0.txt", "r");
    `endif
    // Following three lines are to remove the first three comment lines of the file
    x_scan_file = $fscanf(x_file,"%s", captured_data);
    x_scan_file = $fscanf(x_file,"%s", captured_data);
    x_scan_file = $fscanf(x_file,"%s", captured_data);
    `ifdef ACT_2BIT
      x1_scan_file = $fscanf(x1_file,"%s", captured_data);
      x1_scan_file = $fscanf(x1_file,"%s", captured_data);
      x1_scan_file = $fscanf(x1_file,"%s", captured_data);
    `endif

    //////// Reset /////////
    #0.5 clk = 1'b0;   reset = 1;
    #0.5 clk = 1'b1;

    for (i=0; i<10 ; i=i+1) begin
      #0.5 clk = 1'b0;
      #0.5 clk = 1'b1;
    end

    #0.5 clk = 1'b0;   reset = 0;
    #0.5 clk = 1'b1;

    #0.5 clk = 1'b0;
    #0.5 clk = 1'b1;
    /////////////////////////

    /////// Activation data writing to memory ///////
    $display("## Activation.txt Writing to X_MEM Start ");
    for (t=0; t<len_nij; t=t+1) begin
      #0.5 clk = 1'b0;
      `ifdef ACT_2BIT
        x_scan_file = $fscanf(x_file,"%16b", tile0_data);
        x1_scan_file = $fscanf(x1_file,"%16b", tile1_data);
        // Interleave: tile1 in MSB, tile0 in LSB, 2-bit step
        // Format: [t1_7[1:0] t0_7[1:0] t1_6[1:0] t0_6[1:0] ... t1_0[1:0] t0_0[1:0]]
        D_xmem[31:30] = tile1_data[15:14];  // t1_7
        D_xmem[29:28] = tile0_data[15:14];  // t0_7
        D_xmem[27:26] = tile1_data[13:12];  // t1_6
        D_xmem[25:24] = tile0_data[13:12];  // t0_6
        D_xmem[23:22] = tile1_data[11:10];  // t1_5
        D_xmem[21:20] = tile0_data[11:10];  // t0_5
        D_xmem[19:18] = tile1_data[9:8];    // t1_4
        D_xmem[17:16] = tile0_data[9:8];    // t0_4
        D_xmem[15:14] = tile1_data[7:6];    // t1_3
        D_xmem[13:12] = tile0_data[7:6];    // t0_3
        D_xmem[11:10] = tile1_data[5:4];    // t1_2
        D_xmem[9:8]   = tile0_data[5:4];    // t0_2
        D_xmem[7:6]   = tile1_data[3:2];    // t1_1
        D_xmem[5:4]   = tile0_data[3:2];    // t0_1
        D_xmem[3:2]   = tile1_data[1:0];     // t1_0
        D_xmem[1:0]   = tile0_data[1:0];     // t0_0
      `else
        x_scan_file = $fscanf(x_file,"%32b", D_xmem);
      `endif
      WEN_xmem = 0; CEN_xmem = 0; if (t>0) A_xmem = A_xmem + 1;
      #0.5 clk = 1'b1;
      // $display("Writing to address %5d", A_xmem);
    end

    #0.5 clk = 1'b0;  WEN_xmem = 1;  CEN_xmem = 1; A_xmem = 0;
    #0.5 clk = 1'b1;

    $fclose(x_file);
    `ifdef ACT_2BIT
      $fclose(x1_file);
    `endif
    /////////////////////////////////////////////////
    $display("## Activation.txt Writing to X_MEM End");


    for (kij=0; kij<9; kij=kij+1) begin  // kij loop
    kij_SFUctrl = kij;
    $display("## kij=%1d", kij);

    `ifdef ACT_2BIT
      case (kij)
        0: begin w_file_name = "../datafiles/ws2bit/weight_itile0_otile0_kij0.txt"; w1_file_name = "../datafiles/ws2bit/weight_itile1_otile0_kij0.txt"; end
        1: begin w_file_name = "../datafiles/ws2bit/weight_itile0_otile0_kij1.txt"; w1_file_name = "../datafiles/ws2bit/weight_itile1_otile0_kij1.txt"; end
        2: begin w_file_name = "../datafiles/ws2bit/weight_itile0_otile0_kij2.txt"; w1_file_name = "../datafiles/ws2bit/weight_itile1_otile0_kij2.txt"; end
        3: begin w_file_name = "../datafiles/ws2bit/weight_itile0_otile0_kij3.txt"; w1_file_name = "../datafiles/ws2bit/weight_itile1_otile0_kij3.txt"; end
        4: begin w_file_name = "../datafiles/ws2bit/weight_itile0_otile0_kij4.txt"; w1_file_name = "../datafiles/ws2bit/weight_itile1_otile0_kij4.txt"; end
        5: begin w_file_name = "../datafiles/ws2bit/weight_itile0_otile0_kij5.txt"; w1_file_name = "../datafiles/ws2bit/weight_itile1_otile0_kij5.txt"; end
        6: begin w_file_name = "../datafiles/ws2bit/weight_itile0_otile0_kij6.txt"; w1_file_name = "../datafiles/ws2bit/weight_itile1_otile0_kij6.txt"; end
        7: begin w_file_name = "../datafiles/ws2bit/weight_itile0_otile0_kij7.txt"; w1_file_name = "../datafiles/ws2bit/weight_itile1_otile0_kij7.txt"; end
        8: begin w_file_name = "../datafiles/ws2bit/weight_itile0_otile0_kij8.txt"; w1_file_name = "../datafiles/ws2bit/weight_itile1_otile0_kij8.txt"; end
      endcase
    `else
      case (kij)
        0: w_file_name = "../datafiles/ws4bit/weight_itile0_otile0_kij0.txt";
        1: w_file_name = "../datafiles/ws4bit/weight_itile0_otile0_kij1.txt";
        2: w_file_name = "../datafiles/ws4bit/weight_itile0_otile0_kij2.txt";
        3: w_file_name = "../datafiles/ws4bit/weight_itile0_otile0_kij3.txt";
        4: w_file_name = "../datafiles/ws4bit/weight_itile0_otile0_kij4.txt";
        5: w_file_name = "../datafiles/ws4bit/weight_itile0_otile0_kij5.txt";
        6: w_file_name = "../datafiles/ws4bit/weight_itile0_otile0_kij6.txt";
        7: w_file_name = "../datafiles/ws4bit/weight_itile0_otile0_kij7.txt";
        8: w_file_name = "../datafiles/ws4bit/weight_itile0_otile0_kij8.txt";
      endcase
    `endif
    

    w_file = $fopen(w_file_name, "r");
    if (w_file == 0) begin
      $display("ERROR: cannot open file %s", w_file_name);
      $finish;
    end
    // Following three lines are to remove the first three comment lines of the file
    w_scan_file = $fscanf(w_file,"%s", captured_data);
    w_scan_file = $fscanf(w_file,"%s", captured_data);
    w_scan_file = $fscanf(w_file,"%s", captured_data);
    `ifdef ACT_2BIT
      w1_file = $fopen(w1_file_name, "r");
      if (w1_file == 0) begin
        $display("ERROR: cannot open file %s", w1_file_name);
        $finish;
      end
      // Following three lines are to remove the first three comment lines of the file
      w1_scan_file = $fscanf(w1_file,"%s", captured_data);
      w1_scan_file = $fscanf(w1_file,"%s", captured_data);
      w1_scan_file = $fscanf(w1_file,"%s", captured_data);
    `endif

    #0.5 clk = 1'b0;   reset = 1;
    #0.5 clk = 1'b1; 

    for (i=0; i<10 ; i=i+1) begin
      #0.5 clk = 1'b0;
      #0.5 clk = 1'b1;  
    end

    #0.5 clk = 1'b0;   reset = 0;
    #0.5 clk = 1'b1; 

    #0.5 clk = 1'b0;   
    #0.5 clk = 1'b1;   





    /////// Kernel data writing to memory ///////
    $display("   weight.txt Writing to X_MEM Start ");
    A_xmem = 11'b10000000000;
    `ifdef ACT_2BIT
      for (t=0; t<mac_col; t=t+1) begin  
      // Write tile1 weight data first
        #0.5 clk = 1'b0;  w1_scan_file = $fscanf(w1_file,"%32b", D_xmem); WEN_xmem = 0; CEN_xmem = 0; if (t>0) A_xmem = A_xmem + 1; 
        #0.5 clk = 1'b1;  
      // Write tile0 weight data
        #0.5 clk = 1'b0;  w_scan_file = $fscanf(w_file,"%32b", D_xmem); WEN_xmem = 0; CEN_xmem = 0; A_xmem = A_xmem + 1; 
        #0.5 clk = 1'b1;  
      end
    `else
      for (t=0; t<mac_col; t=t+1) begin  
        #0.5 clk = 1'b0;  w_scan_file = $fscanf(w_file,"%32b", D_xmem); WEN_xmem = 0; CEN_xmem = 0; if (t>0) A_xmem = A_xmem + 1; 
        #0.5 clk = 1'b1;  
      end
    `endif

    #0.5 clk = 1'b0;  WEN_xmem = 1;  CEN_xmem = 1; A_xmem = 0;
    #0.5 clk = 1'b1; 
    $display("   weight.txt Writing to X_MEM End");
    /////////////////////////////////////

    /////// Kernel data writing to L0 ///////
    $display("   Weight Feeding to L0FIFO Start");
    A_xmem = 11'b10000000000;
    `ifdef ACT_2BIT
      for (t=0; t<mac_col; t=t+1) begin  
        // Feed tile1 weight data to L0FIFO first
        #0.5 clk = 1'b0; WEN_xmem = 1; CEN_xmem = 0; inst_w = 2'b01; if (t>0) A_xmem = A_xmem + 1;
        #0.5 clk = 1'b1;  
        // Feed tile0 weight data to L0FIFO
        #0.5 clk = 1'b0; WEN_xmem = 1; CEN_xmem = 0; inst_w = 2'b01; A_xmem = A_xmem + 1;
        #0.5 clk = 1'b1;  
      end
    `else
      for (t=0; t<mac_col; t=t+1) begin  
        #0.5 clk = 1'b0; WEN_xmem = 1; CEN_xmem = 0; inst_w = 2'b01; if (t>0) A_xmem = A_xmem + 1;
        #0.5 clk = 1'b1;  
      end  
    `endif

    $display("   Weight Feeding to L0FIFO End.");
    #0.5 clk = 1'b0;  inst_w = 2'b00; WEN_xmem = 1; CEN_xmem = 1;
    #0.5 clk = 1'b1;  
    /////////////////////////////////////



    /////// Activation data writing to L0 ///////
    $display("   Activation Feeding to L0FIFO Start");
    A_xmem = 11'b00000000000;
    for (t=0; t<len_nij; t=t+1) begin  
      #0.5 clk = 1'b0; WEN_xmem = 1; CEN_xmem = 0; inst_w = 2'b10; if (t>0) A_xmem = A_xmem + 1;
      #0.5 clk = 1'b1;  
    end
    /////////////////////////////////////



    /////// Execution ///////
    ////// provide some intermission to clear up the activation running ///
    #0.5 clk = 1'b0;  WEN_xmem = 1; CEN_xmem = 1; inst_w = 2'b00;
    #0.5 clk = 1'b1;  
    $display("   Activation Feeding End. Waiting for PSUM flowing\n");
    for (i=0; i<30 ; i=i+1) begin
      #0.5 clk = 1'b0;
      #0.5 clk = 1'b1;  
    end
    /////////////////////////////////////
    /////////////////////////////////////
    `ifdef ACT_2BIT
      $fclose(w_file);
      $fclose(w1_file);
    `else
      $fclose(w_file);
    `endif
  end  // end of kij loop

    $display("## Conv End. Waiting for ReLU\n");
    for (i=0; i<20 ; i=i+1) begin
      #0.5 clk = 1'b0;
      #0.5 clk = 1'b1;  
    end


  $display("############ Output Verification Start #############"); 
  `ifdef ACT_2BIT
    // out_file = $fopen("../datafiles/ws2bit/output.txt", "r");
    out_file = $fopen("../datafiles/ws2bit/expected_output_from_psum_binary.txt", "r");
  `else
    out_file = $fopen("../datafiles/ws4bit/out.txt", "r");
  `endif  
  // Following three lines are to remove the first three comment lines of the file
  out_scan_file = $fscanf(out_file,"%s", answer); 
  out_scan_file = $fscanf(out_file,"%s", answer); 
  out_scan_file = $fscanf(out_file,"%s", answer); 
  error = 0;
  

  #0.5 clk = 1'b0; readout_start = 1'b1;
  #0.5 clk = 1'b1; // Clock Pos Edge1: core.v notified readout_start

  #0.5 clk = 1'b0; readout_start = 1'b0;
  #0.5 clk = 1'b1; // Clock Pos Edge2: readout port starts output

  for (i=0; i<len_onij; i=i+1) begin 
    #0.5 clk = 1'b0; // Clock Neg Edge

    out_scan_file = $fscanf(out_file,"%128b", answer); // reading from out file to answer
    if (readout == answer)
      $display("%2d-th output featuremap Data matched! :D", i); 
    else begin
      $display("%2d-th output featuremap Data ERROR!!", i); 
      $display("sfpout: %8h", readout);
      $display("answer: %8h", answer[psum_bw*mac_col-1:0]);
      error = 1;
    end
    #0.5 clk = 1'b1; // Clock Pos Edge3: readout port output next
  end

    if (error == 0) begin
      $display("############ No error detected ##############");
      $display("########### Project Completed !! ############");
    end


  //////////////////////////////////
  for (t=0; t<10; t=t+1) begin  
    #0.5 clk = 1'b0;  
    #0.5 clk = 1'b1;  
  end
  #10 $finish;

  end


endmodule



