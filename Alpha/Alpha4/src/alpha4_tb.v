// Created by prof. Mingu Kang @VVIP Lab in UCSD ECE department
// Please do not spread this code without permission 
`timescale 1ns/1ps

module alpha4_tb;

parameter bw = 4;
parameter psum_bw = 16;
parameter len_kij = 9;
parameter len_onij = 16;
parameter col = 8;
parameter row = 8;
parameter len_nij = 36;

reg clk = 0;
reg reset = 1;

// Readout (for output.txt verification. tb has no access to PSUM mem)
reg readout_start;
wire [psum_bw*col-1:0] readout;

// Control Signal for SFU.
reg [3:0] kij_SFUctrl;
reg                     ofifo_valid;
reg [psum_bw*col-1 : 0] ofifo_data;
reg [psum_bw*col-1 : 0] bias;

// Connect PMEM and SFU.
wire [psum_bw*col-1:0]  Q_pmem;
wire ren_pmem;
wire wen_pmem;
wire [3:0]              w_A_pmem;
wire [3:0]              r_A_pmem;
wire [psum_bw*col-1:0]  D_pmem;

// Misc 
reg [8*30:1] stringvar;
reg [8*50:1] psum_file_name;
reg [psum_bw*col-1:0] answer;
integer psum_file, psum_scan_file ; // file_handler
integer out_file, out_scan_file ; // file_handler
integer captured_data; 
integer t, i, j, k, kij, psum_int;
integer error;






SFU#(.psum_bw(psum_bw), .col(col)) SFU_instance(
  .clk(clk),
  .reset(reset),
  // sense signal from ofifo and output ctrl
  .ofifo_valid(ofifo_valid),
  .ofifo_data(ofifo_data),
  .bias(bias),
  // data in and ctrl signal for PSUM SRAM
  .Q_pmem(Q_pmem),  
  .ren_pmem(ren_pmem),
  .wen_pmem(wen_pmem),
  .r_A_pmem(r_A_pmem), // o_nij = 0~15
  .w_A_pmem(w_A_pmem),
  .D_pmem(D_pmem),
  // kernel index
  .kij(kij_SFUctrl),
  // readout phase
  .readout_start(readout_start), // trigger for output stage
  .readout(readout)        // 16*8b
);

sram_128b_w16_RW P_MEM_instance(
  .CLK(clk),
  .ren(ren_pmem), 
  .wen(wen_pmem), 
  .w_A(w_A_pmem),
  .r_A(r_A_pmem),
  .D(D_pmem), 
  .Q(Q_pmem)
);





initial begin 
  kij_SFUctrl    = 0;
  readout_start = 0;

  bias = {(psum_bw*col){1'b0}};

  $dumpfile("alpha4_tb.vcd");
  $dumpvars(0, alpha4_tb);
  


  /////// Weight data reading to w_vector reg ///////
  for (kij=0; kij<9; kij=kij+1) begin  // kij loop

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








    kij_SFUctrl = kij;
    ofifo_valid = 1'b0;
    $display("## kij=%1d", kij);

    case(kij)
      0: psum_file_name = "golden/psum_kij0.txt";
      1: psum_file_name = "golden/psum_kij1.txt";
      2: psum_file_name = "golden/psum_kij2.txt";
      3: psum_file_name = "golden/psum_kij3.txt";
      4: psum_file_name = "golden/psum_kij4.txt";
      5: psum_file_name = "golden/psum_kij5.txt";
      6: psum_file_name = "golden/psum_kij6.txt";
      7: psum_file_name = "golden/psum_kij7.txt";
      8: psum_file_name = "golden/psum_kij8.txt";
    endcase

    psum_file = $fopen(psum_file_name, "r");
    if (psum_file == 0) begin
      $display("ERROR: cannot open file %s", psum_file_name);
      $finish;
    end
    // Following three lines are to remove the first three comment lines of the file
    psum_scan_file = $fscanf(psum_file,"%s", captured_data);
    psum_scan_file = $fscanf(psum_file,"%s", captured_data);
    psum_scan_file = $fscanf(psum_file,"%s", captured_data);


    $display("   psum.txt Reading Start ");
    for (t=0; t<len_nij; t=t+1) begin   

        #0.5 clk = 1'b0;   
        psum_scan_file = $fscanf(psum_file,"%128b", ofifo_data);
        ofifo_valid = 1'b1;

        // ###############################
        #0.5 clk = 1'b1; 





    end

    ofifo_valid = 1'b0;
    $display("   psum.txt Reading End");
    $fclose(psum_file);


    //////// Idle /////////
    for (i=0; i<30 ; i=i+1) begin
      #0.5 clk = 1'b0;
      #0.5 clk = 1'b1;  
    end


  end
  

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


  // Output is MaxPooled!
  for (i=0; i<4; i=i+1) begin 
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




