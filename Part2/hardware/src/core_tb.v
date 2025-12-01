// Created by prof. Mingu Kang @VVIP Lab in UCSD ECE department
// Please do not spread this code without permission 
`timescale 1ns/1ps

module core_tb;

parameter bw = 4;
parameter psum_bw = 16;
parameter len_kij = 9;
parameter len_onij = 16;
parameter col = 8;
parameter row = 8;
parameter len_nij = 36;

reg clk = 0;
reg reset = 1;

// High level instrucion & data for mac array.
reg [1:0]  inst_w; 
reg [bw*row-1:0] D_xmem;

// X_MEM Ctrl(for activation.txt and weight.txt writing)
reg CEN_xmem = 1;
reg WEN_xmem = 1;
reg [10:0] A_xmem = 0;

// Readout (for output.txt verification. tb has no access to PSUM mem)
reg readout_start;
wire [psum_bw*col-1:0] readout;

// Control Signal for SFU. I hope to get rid of this in the future versions.
reg [3:0] kij_SFUctrl;


// Misc 
reg [8*30:1] stringvar;
reg [8*50:1] w_file_name;
reg [psum_bw*col-1:0] answer;
integer x_file, x_scan_file ; // file_handler
integer w_file, w_scan_file ; // file_handler
integer out_file, out_scan_file ; // file_handler
integer captured_data; 
integer t, i, j, k, kij;
integer error;


core #(.bw(bw), .psum_bw(psum_bw), .col(col), .row(row)) core_instance (
	.clk(clk), 
  .reset(reset),
  .inst_w(inst_w),     // high level instructions from TB
  .CEN_xmem(CEN_xmem), // x_mem ctrls from TB
  .WEN_xmem(WEN_xmem),
  .A_xmem(A_xmem),
  .D_xmem(D_xmem), 
  .kij(kij_SFUctrl),        // SFU control
  .readout_start(readout_start), // Output to TB
  .readout(readout)
); 


initial begin 
  inst_w   = 0; 
  D_xmem   = 0;
  CEN_xmem = 1;
  WEN_xmem = 1;
  A_xmem   = 0;
  kij_SFUctrl    = 0;
  readout_start = 0;

  $dumpfile("core_tb.vcd");
  $dumpvars(0, core_tb);

  x_file = $fopen("golden/activation_tile0.txt", "r");
  // Following three lines are to remove the first three comment lines of the file
  x_scan_file = $fscanf(x_file,"%s", captured_data);
  x_scan_file = $fscanf(x_file,"%s", captured_data);
  x_scan_file = $fscanf(x_file,"%s", captured_data);

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
    #0.5 clk = 1'b0;  x_scan_file = $fscanf(x_file,"%32b", D_xmem); WEN_xmem = 0; CEN_xmem = 0; if (t>0) A_xmem = A_xmem + 1;
    #0.5 clk = 1'b1;   
    // $display("Writing to address %5d", A_xmem);
  end
  
  #0.5 clk = 1'b0;  WEN_xmem = 1;  CEN_xmem = 1; A_xmem = 0;
  #0.5 clk = 1'b1; 

  $fclose(x_file);
  /////////////////////////////////////////////////
  $display("## Activation.txt Writing to X_MEM End");


  for (kij=0; kij<9; kij=kij+1) begin  // kij loop
    kij_SFUctrl = kij;
    $display("## kij=%1d", kij);

    case(kij)
     0: w_file_name = "golden/weight_itile0_otile0_kij0.txt";
     1: w_file_name = "golden/weight_itile0_otile0_kij1.txt";
     2: w_file_name = "golden/weight_itile0_otile0_kij2.txt";
     3: w_file_name = "golden/weight_itile0_otile0_kij3.txt";
     4: w_file_name = "golden/weight_itile0_otile0_kij4.txt";
     5: w_file_name = "golden/weight_itile0_otile0_kij5.txt";
     6: w_file_name = "golden/weight_itile0_otile0_kij6.txt";
     7: w_file_name = "golden/weight_itile0_otile0_kij7.txt";
     8: w_file_name = "golden/weight_itile0_otile0_kij8.txt";
    endcase
    

    w_file = $fopen(w_file_name, "r");
    if (w_file == 0) begin
      $display("ERROR: cannot open file %s", w_file_name);
      $finish;
    end
    // Following three lines are to remove the first three comment lines of the file
    w_scan_file = $fscanf(w_file,"%s", captured_data);
    w_scan_file = $fscanf(w_file,"%s", captured_data);
    w_scan_file = $fscanf(w_file,"%s", captured_data);

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
    for (t=0; t<col; t=t+1) begin  
      #0.5 clk = 1'b0;  w_scan_file = $fscanf(w_file,"%32b", D_xmem); WEN_xmem = 0; CEN_xmem = 0; if (t>0) A_xmem = A_xmem + 1; 
      #0.5 clk = 1'b1;  
    end

    #0.5 clk = 1'b0;  WEN_xmem = 1;  CEN_xmem = 1; A_xmem = 0;
    #0.5 clk = 1'b1; 
    $display("   weight.txt Writing to X_MEM End");
    /////////////////////////////////////

    /////// Kernel data writing to L0 ///////
    $display("   Weight Feeding to L0FIFO Start");
    A_xmem = 11'b10000000000;
    for (t=0; t<col; t=t+1) begin  
      #0.5 clk = 1'b0; WEN_xmem = 1; CEN_xmem = 0; inst_w = 2'b01; if (t>0) A_xmem = A_xmem + 1;
      #0.5 clk = 1'b1;  
    end  
    

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
  end  // end of kij loop




  $display("############ Output Verification Start #############"); 
  out_file = $fopen("golden/out.txt", "r");  
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
      $display("sfpout: %128b", readout);
      $display("answer: %128b", answer);
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



