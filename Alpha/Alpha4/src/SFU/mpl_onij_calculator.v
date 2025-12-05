module mpl_onij_calculator (
    input  [3:0]  order, // 0~15
    output [3:0]  o_nij, // 0~15
    output [1:0]  mpl_onij // 0~15
);

    // order 0~3 is the mpl_onij=2'b00
    // order 4~7 is the mpl_onij=2'b01
    // order 8~11 is the mpl_onij=2'b10
    // order 12~15 is the mpl_onij=2'b11
    assign mpl_onij = order[3:2];
    assign o_nij    = {order[3], order[1], order[2], order[0]};
    // This simpling wiring pattern is found by K-map, which is very beautiful


    
endmodule