`timescale 1ns / 1ps

/*
 * Module: simple_gate_axi_stream
 * Description: AXI4-Stream compliant hardware coprocessor for NAFNet's 
 * SimpleGate activation function. Accepts dual INT8 feature streams (s_axis) 
 * and outputs an INT16 gated feature stream (m_axis) with full backpressure logic.
 */

module simple_gate_axi_stream #(
    parameter DATA_WIDTH = 8
)(
    input wire clk,
    input wire aresetn, // Active-low synchronous reset

    // --- AXI4-Stream Slave Interface (Inputs) ---
    input  wire signed [DATA_WIDTH-1:0]  s_axis_tdata1,
    input  wire signed [DATA_WIDTH-1:0]  s_axis_tdata2,
    input  wire                          s_axis_tvalid,
    output wire                          s_axis_tready,

    // --- AXI4-Stream Master Interface (Outputs) ---
    output reg  signed [(2*DATA_WIDTH)-1:0] m_axis_tdata,
    output reg                           m_axis_tvalid,
    input  wire                          m_axis_tready
);

    // Ready whenever Master stream downstream is ready to accept data
    assign s_axis_tready = m_axis_tready || !m_axis_tvalid;

    always @(posedge clk or negedge aresetn) begin
        if (!aresetn) begin
            m_axis_tdata  <= 0;
            m_axis_tvalid <= 1'b0;
        end else begin
            if (s_axis_tready) begin
                if (s_axis_tvalid) begin
                    m_axis_tdata  <= s_axis_tdata1 * s_axis_tdata2;
                    m_axis_tvalid <= 1'b1;
                end else begin
                    m_axis_tvalid <= 1'b0;
                end
            end
        end
    end

endmodule
