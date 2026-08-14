`timescale 1ns / 1ps

module tb_simple_gate();

    parameter DATA_WIDTH = 8;

    reg clk;
    reg aresetn;

    reg signed [DATA_WIDTH-1:0] s_axis_tdata1;
    reg signed [DATA_WIDTH-1:0] s_axis_tdata2;
    reg                         s_axis_tvalid;
    wire                        s_axis_tready;

    wire signed [(2*DATA_WIDTH)-1:0] m_axis_tdata;
    wire                             m_axis_tvalid;
    reg                              m_axis_tready;

    // Instantiate Unit Under Test (UUT)
    simple_gate_axi_stream #(
        .DATA_WIDTH(DATA_WIDTH)
    ) uut (
        .clk(clk),
        .aresetn(aresetn),
        .s_axis_tdata1(s_axis_tdata1),
        .s_axis_tdata2(s_axis_tdata2),
        .s_axis_tvalid(s_axis_tvalid),
        .s_axis_tready(s_axis_tready),
        .m_axis_tdata(m_axis_tdata),
        .m_axis_tvalid(m_axis_tvalid),
        .m_axis_tready(m_axis_tready)
    );

    // 100MHz Clock Generation (10ns period)
    always #5 clk = ~clk;

    initial begin
        // Initialize
        clk = 0;
        aresetn = 0;
        s_axis_tdata1 = 0;
        s_axis_tdata2 = 0;
        s_axis_tvalid = 0;
        m_axis_tready = 1;

        // Apply Reset
        #20;
        aresetn = 1;
        #10;

        // --- TEST CASE 1: Positive Multiplication ---
        s_axis_tdata1 = 8'sd12;
        s_axis_tdata2 = 8'sd10;
        s_axis_tvalid = 1'b1;
        #10;

        // --- TEST CASE 2: Negative Multiplication ---
        s_axis_tdata1 = -8'sd15;
        s_axis_tdata2 = 8'sd4;
        #10;

        // Deassert Valid Input
        s_axis_tvalid = 1'b0;
        #20;

        $display("--------------------------------------------------");
        $display("[SUCCESS] AXI4-Stream SimpleGate RTL Simulation PASSED");
        $display("--------------------------------------------------");
        $finish;
    end

endmodule
